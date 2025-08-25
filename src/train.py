import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Global plotting style for paper-quality PDFs
# ------------------------------
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(style="whitegrid", font_scale=1.1)

# ------------------------------
# Paths and utilities
# ------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_images_dir() -> str:
    d = os.path.join(".research", "iteration1", "images")
    ensure_dir(d)
    return d


def get_models_dir() -> str:
    d = os.path.join("models")
    ensure_dir(d)
    return d


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_ctx(device: torch.device):
    if device.type == 'cuda':
        return torch.cuda.amp.autocast()
    else:
        from contextlib import nullcontext
        return nullcontext()


# ------------------------------
# Synthetic data generators (fast, deterministic if seed set)
# ------------------------------

def gen_checker(batch: int, size: int, squares: int = 4) -> torch.Tensor:
    x = torch.zeros(batch, 3, size, size)
    block = size // squares
    for b in range(batch):
        img = torch.zeros(1, size, size)
        for i in range(squares):
            for j in range(squares):
                if (i + j) % 2 == 0:
                    img[:, i * block:(i + 1) * block, j * block:(j + 1) * block] = 1.0
        x[b] = img.repeat(3, 1, 1)
    return x * 2 - 1


def gen_stripes(batch: int, size: int, orientation: str = 'h') -> torch.Tensor:
    x = torch.zeros(batch, 3, size, size)
    for b in range(batch):
        img = torch.zeros(1, size, size)
        if orientation == 'h':
            img[:, ::2, :] = 1.0
        else:
            img[:, :, ::2] = 1.0
        x[b] = img.repeat(3, 1, 1)
    return x * 2 - 1


def gen_gaussian_blobs(batch: int, size: int, n_blobs: int = 3) -> torch.Tensor:
    x = torch.zeros(batch, 3, size, size)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    grid = torch.stack([yy, xx], dim=0)
    for b in range(batch):
        img = torch.zeros(1, size, size)
        for _ in range(n_blobs):
            cy = random.uniform(-0.8, 0.8)
            cx = random.uniform(-0.8, 0.8)
            sigma = random.uniform(0.05, 0.2)
            g = torch.exp(-((grid[0] - cy) ** 2 + (grid[1] - cx) ** 2) / (2 * sigma ** 2))
            img += g.unsqueeze(0)
        img = img.clamp(0, 1)
        x[b] = img.repeat(3, 1, 1)
    return x * 2 - 1


def gen_radial_gradient(batch: int, size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    img = (1 - rr).clamp(0, 1)
    x = img.unsqueeze(0).repeat(batch, 3, 1, 1)
    return x * 2 - 1


class SyntheticDataset(Dataset):
    def __init__(self, size: int, steps: int, patterns: Tuple[str, ...] = ("checker", "blobs", "stripes_h", "radial")):
        super().__init__()
        self.size = size
        self.steps = steps
        self.patterns = list(patterns)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        patt = self.patterns[idx % len(self.patterns)]
        if patt == 'checker':
            x = gen_checker(1, self.size)
        elif patt == 'blobs':
            x = gen_gaussian_blobs(1, self.size)
        elif patt == 'stripes_h':
            x = gen_stripes(1, self.size, 'h')
        elif patt == 'stripes_v':
            x = gen_stripes(1, self.size, 'v')
        elif patt == 'radial':
            x = gen_radial_gradient(1, self.size)
        else:
            x = torch.randn(1, 3, self.size, self.size).clamp(-1, 1)
        return x[0]


# ------------------------------
# RevCol-Diff model components
# ------------------------------

class InplaceGroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class QuantizedSineEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=device) / half)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        maxv = emb.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)
        q = torch.clamp((emb / maxv) * 127.0, -127, 127).round().to(torch.int8)
        deq = q.float() / 127.0 * maxv
        return deq


class ColumnAdapter(nn.Module):
    def __init__(self, C: int, K: int):
        super().__init__()
        assert 0 < K <= C
        self.C, self.K = C, K
        self.proj = nn.Conv2d(C, K, kernel_size=1, bias=False)
        self.recon = nn.Conv2d(K, C, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.randn(K, C)
            q, _ = torch.linalg.qr(w.T, mode='reduced')  # C x K
            self.proj.weight.copy_(q.T.unsqueeze(-1).unsqueeze(-1))
            pinv = torch.linalg.pinv(q.T)
            self.recon.weight.copy_(pinv.T.unsqueeze(-1).unsqueeze(-1))

    def to_columns(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def from_columns(self, z: torch.Tensor) -> torch.Tensor:
        return self.recon(z)


class RevBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, Fmod: nn.Module, Gmod: nn.Module):
        with torch.no_grad():
            y1 = x1 + Fmod(x2)
            y2 = x2 + Gmod(y1)
        ctx.save_for_backward(y1, y2)
        ctx.Fmod = Fmod
        ctx.Gmod = Gmod
        return y1, y2

    @staticmethod
    def backward(ctx, gy1: torch.Tensor, gy2: torch.Tensor):
        y1, y2 = ctx.saved_tensors
        Fmod = ctx.Fmod
        Gmod = ctx.Gmod
        with torch.no_grad():
            x2 = y2 - Gmod(y1)
            x1 = y1 - Fmod(x2)
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        with torch.enable_grad():
            y1_rec = x1 + Fmod(x2)
            y2_rec = x2 + Gmod(y1_rec)
            torch.autograd.backward([y1_rec, y2_rec], [gy1, gy2])
        return x1.grad, x2.grad, None, None


class DSConvBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            InplaceGroupNorm(d),
            nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RevDualStream(nn.Module):
    def __init__(self, C: int, K: int):
        super().__init__()
        assert K % 2 == 0, "K must be even to split into two streams"
        self.col = ColumnAdapter(C, K)
        self.Fmod = DSConvBlock(K // 2)
        self.Gmod = DSConvBlock(K // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.col.to_columns(x)
        z1, z2 = torch.chunk(z, 2, dim=1)
        y1, y2 = RevBlockFunction.apply(z1, z2, self.Fmod, self.Gmod)
        z_out = torch.cat([y1, y2], dim=1)
        x_out = self.col.from_columns(z_out)
        return x_out


class RevColUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, K: int = 16, num_blocks: int = 2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.d1 = nn.ModuleList([RevDualStream(base_ch, K) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([RevDualStream(base_ch * 2, K) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.mid = RevDualStream(base_ch * 4, K)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.u2 = nn.ModuleList([RevDualStream(base_ch * 2, K) for _ in range(num_blocks)])
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.u1 = nn.ModuleList([RevDualStream(base_ch, K) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_conv(x)
        skips = []
        for m in self.d1:
            h = m(h)
        skips.append(h)
        h = self.down1(h)
        for m in self.d2:
            h = m(h)
        skips.append(h)
        h = self.down2(h)
        h = self.mid(h)
        h = self.up2(h) + skips.pop()
        for m in self.u2:
            h = m(h)
        h = self.up1(h) + skips.pop()
        for m in self.u1:
            h = m(h)
        return self.out_conv(h)


# ------------------------------
# Baseline ADM-like U-Net
# ------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(x))
        h = self.conv1(h)
        h = F.silu(self.gn2(h))
        h = self.conv2(h)
        return x + h


class ADMUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_blocks: int = 2, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        from torch.utils.checkpoint import checkpoint as checkpoint_fn
        self._ckpt = checkpoint_fn
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.d1 = nn.ModuleList([ResBlock(base_ch) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([ResBlock(base_ch * 2) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.mid = ResBlock(base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.u2 = nn.ModuleList([ResBlock(base_ch * 2) for _ in range(num_blocks)])
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.u1 = nn.ModuleList([ResBlock(base_ch) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def _apply_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return self._ckpt(block, x)
        else:
            return block(x)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_conv(x)
        skips = []
        for m in self.d1:
            h = self._apply_block(m, h)
        skips.append(h)
        h = self.down1(h)
        for m in self.d2:
            h = self._apply_block(m, h)
        skips.append(h)
        h = self.down2(h)
        h = self._apply_block(self.mid, h)
        h = self.up2(h) + skips.pop()
        for m in self.u2:
            h = self._apply_block(m, h)
        h = self.up1(h) + skips.pop()
        for m in self.u1:
            h = self._apply_block(m, h)
        return self.out_conv(h)


# ------------------------------
# Diffusion wrapper (v-pred, cosine schedule)
# ------------------------------

class DiffusionWrapper(nn.Module):
    def __init__(self, net: nn.Module, num_steps: int = 1000):
        super().__init__()
        self.net = net
        self.num_steps = num_steps

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.num_steps, (B,), device=device)
        s = 0.008
        t_ = (t.float() + 0.5) / self.num_steps
        alpha_bar = torch.cos((t_ + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar.view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
        v = torch.sqrt(alpha_bar) * eps - torch.sqrt(1 - alpha_bar) * x0
        v_pred = self.net(x_t, t)
        return F.mse_loss(v_pred, v)


# ------------------------------
# Training configuration and loop
# ------------------------------

@dataclass
class TrainConfig:
    model: str = "revcol"  # choices: revcol, adm, adm_ckpt
    image_size: int = 32
    batch_size: int = 8
    steps: int = 200
    lr: float = 1e-3
    num_workers: int = 2
    base_ch: int = 64
    K: int = 16
    num_blocks: int = 2
    patterns_train: Tuple[str, ...] = ("checker", "blobs", "stripes_h", "radial")
    patterns_val: Tuple[str, ...] = ("checker", "stripes_v", "blobs", "noise")
    amp: bool = True


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model == 'revcol':
        return RevColUNet(in_ch=3, base_ch=cfg.base_ch, K=cfg.K, num_blocks=cfg.num_blocks)
    elif cfg.model == 'adm':
        return ADMUNet(in_ch=3, base_ch=cfg.base_ch, num_blocks=cfg.num_blocks, use_checkpoint=False)
    elif cfg.model == 'adm_ckpt':
        return ADMUNet(in_ch=3, base_ch=cfg.base_ch, num_blocks=cfg.num_blocks, use_checkpoint=True)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def make_loader(size: int, batch: int, steps: int, patterns: Tuple[str, ...], nw: int) -> DataLoader:
    ds = SyntheticDataset(size=size, steps=steps, patterns=patterns)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())


def train_one(cfg: TrainConfig, device: torch.device) -> Dict[str, Any]:
    images_dir = get_images_dir()
    models_dir = get_models_dir()

    model = build_model(cfg)
    net = DiffusionWrapper(model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)

    train_loader = make_loader(cfg.image_size, cfg.batch_size, cfg.steps, cfg.patterns_train, cfg.num_workers)
    val_loader = make_loader(cfg.image_size, cfg.batch_size, max(10, cfg.steps // 5), cfg.patterns_val, cfg.num_workers)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

    train_losses: List[float] = []
    val_losses_snap: List[float] = []

    net.train()
    for step, xb in enumerate(train_loader, start=1):
        xb = xb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            loss = net(xb)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        train_losses.append(float(loss.item()))

        if step % max(1, cfg.steps // 4) == 0:
            print(f"[Train] step {step}/{cfg.steps} loss={loss.item():.4f}")

        # periodic validation snapshot
        if step % max(1, cfg.steps // len(val_loader)) == 0:
            with torch.no_grad():
                vb = next(iter(val_loader)).to(device)
                vloss = float(net(vb).item())
                val_losses_snap.append(vloss)

        if step >= cfg.steps:
            break

    # final validation
    net.eval()
    val_losses = []
    with torch.no_grad():
        for vb in val_loader:
            vb = vb.to(device)
            val_losses.append(float(net(vb).item()))
    val_mean = float(np.mean(val_losses)) if len(val_losses) > 0 else float('nan')

    # save model
    ckpt_path = os.path.join(models_dir, f"{cfg.model}_is{cfg.image_size}_bs{cfg.batch_size}_steps{cfg.steps}.pt")
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, ckpt_path)

    # plots
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label=cfg.model)
    plt.title('Training loss (v-pred MSE)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"training_loss_{cfg.model}.pdf"), bbox_inches='tight')
    plt.close()

    if len(val_losses_snap) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(val_losses_snap, marker='o', label=f"{cfg.model} (val)")
        plt.title('Validation loss snapshots')
        plt.xlabel('Snapshot idx')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f"validation_loss_{cfg.model}.pdf"), bbox_inches='tight')
        plt.close()

    print(f"[Train] Completed. Final val_loss={val_mean:.4f}. Model saved to {ckpt_path}")

    return {
        'train_losses': train_losses,
        'val_mean': val_mean,
        'val_losses_snap': val_losses_snap,
        'ckpt_path': ckpt_path,
    }


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='RevCol-Diff training script')
    p.add_argument('--model', type=str, default='revcol', choices=['revcol', 'adm', 'adm_ckpt'])
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--num_blocks', type=int, default=2)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--patterns_train', type=str, nargs='*', default=["checker", "blobs", "stripes_h", "radial"]) 
    p.add_argument('--patterns_val', type=str, nargs='*', default=["checker", "stripes_v", "blobs", "noise"]) 
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"[Train] Using device: {device}")

    cfg = TrainConfig(
        model=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        base_ch=args.base_ch,
        K=args.K,
        num_blocks=args.num_blocks,
        patterns_train=tuple(args.patterns_train),
        patterns_val=tuple(args.patterns_val),
        amp=(not args.no_amp),
    )

    stats = train_one(cfg, device)
    # Save summary json
    images_dir = get_images_dir()
    summary_path = os.path.join(images_dir, f"train_summary_{cfg.model}.json")
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[Train] Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
