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

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paper-quality PDFs
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


def reset_peak_gpu_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


# ------------------------------
# Import model and diffusion from train.py
# ------------------------------
from train import (
    RevColUNet, ADMUNet, DiffusionWrapper,
    gen_checker, gen_stripes, gen_gaussian_blobs, gen_radial_gradient
)


# ------------------------------
# Synthetic batch sampler for benchmarking
# ------------------------------

def synthetic_benchmark_batch(bs: int, size: int, device: torch.device) -> torch.Tensor:
    patt = random.choice(['checker', 'stripes_h', 'stripes_v', 'blobs', 'radial', 'noise'])
    gens = {
        'checker': lambda: gen_checker(bs, size, squares=4),
        'stripes_h': lambda: gen_stripes(bs, size, 'h'),
        'stripes_v': lambda: gen_stripes(bs, size, 'v'),
        'blobs': lambda: gen_gaussian_blobs(bs, size, n_blobs=2),
        'radial': lambda: gen_radial_gradient(bs, size),
        'noise': lambda: torch.randn(bs, 3, size, size).clamp(-1, 1),
    }
    x = gens[patt]()
    return x.to(device)


@dataclass
class BenchConfig:
    image_size: int
    batch_size: int
    steps: int = 30
    warmup: int = 5
    base_ch: int = 64


def benchmark_model(model: nn.Module, cfg: BenchConfig, device: torch.device) -> Dict[str, float]:
    net = DiffusionWrapper(model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    reset_peak_gpu_mem()

    # warmup
    for _ in range(cfg.warmup):
        x = synthetic_benchmark_batch(cfg.batch_size, cfg.image_size, device)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            loss = net(x)
        loss.backward()
        opt.step()

    # measure
    times = []
    last_loss = None
    for _ in range(cfg.steps):
        x = synthetic_benchmark_batch(cfg.batch_size, cfg.image_size, device)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            loss = net(x)
        loss.backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        last_loss = float(loss.item())

    mem = peak_gpu_mem_mb()
    sec_per_step = sum(times) / len(times)
    img_per_sec = cfg.batch_size / sec_per_step
    return {"peak_mem_mb": mem, "sec_per_step": sec_per_step, "img_per_sec": img_per_sec, "final_loss": last_loss}


# ------------------------------
# E1: Memory/compute trade-off
# ------------------------------

def run_E1_memory_tradeoff(sizes: List[int], batches: List[int], device: torch.device, base_ch_small=64, base_ch_large=96) -> List[Dict[str, Any]]:
    results = []
    print("[E1] Starting memory/compute trade-off experiments...")
    for size in sizes:
        for batch in batches:
            builders = [
                ("RevCol-Diff", lambda: RevColUNet(in_ch=3, base_ch=(base_ch_small if size <= 64 else base_ch_large), K=16, num_blocks=2)),
                ("ADM", lambda: ADMUNet(in_ch=3, base_ch=(base_ch_small if size <= 64 else base_ch_large), num_blocks=2, use_checkpoint=False)),
                ("ADM-ckpt", lambda: ADMUNet(in_ch=3, base_ch=(base_ch_small if size <= 64 else base_ch_large), num_blocks=2, use_checkpoint=True)),
            ]
            for name, builder in builders:
                try:
                    print(f"[E1] size={size}, batch={batch}, model={name}")
                    stats = benchmark_model(builder(), BenchConfig(image_size=size, batch_size=batch, steps=20, warmup=5, base_ch=(base_ch_small if size <= 64 else base_ch_large)), device)
                    row = {"model": name, "size": size, "batch": batch}
                    row.update(stats)
                    print(f"[E1] -> {row}")
                    results.append(row)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"[E1] OOM: model={name}, size={size}, batch={batch}")
                        results.append({"model": name, "size": size, "batch": batch, "peak_mem_mb": None, "img_per_sec": 0.0, "sec_per_step": None, "final_loss": None, "status": "OOM"})
                    else:
                        raise
    return results


def plot_E1(results: List[Dict[str, Any]]):
    import pandas as pd
    df = pd.DataFrame(results)
    images_dir = get_images_dir()

    # Save raw results json
    with open(os.path.join(images_dir, 'E1_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    dfp = df[df["peak_mem_mb"].notnull()].copy()
    if len(dfp) == 0:
        print("[E1] No valid results to plot.")
        return

    for size in sorted(dfp['size'].unique()):
        sub = dfp[dfp['size'] == size]
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=sub, x='batch', y='peak_mem_mb', hue='model', marker='o')
        plt.title(f'E1 Peak GPU memory vs batch (size={size})')
        plt.xlabel('Batch size')
        plt.ylabel('Peak GPU memory (MiB)')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f"E1_peak_memory_tradeoff_size{size}.pdf"), bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.lineplot(data=sub, x='batch', y='img_per_sec', hue='model', marker='o')
        plt.title(f'E1 Throughput vs batch (size={size})')
        plt.xlabel('Batch size')
        plt.ylabel('Images per second')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f"E1_throughput_tradeoff_size{size}.pdf"), bbox_inches="tight")
        plt.close()


# ------------------------------
# E3: Ablation and high-res feasibility
# ------------------------------

def make_rev_variant(base_ch: int, K: int, num_blocks: int) -> nn.Module:
    return RevColUNet(in_ch=3, base_ch=base_ch, K=K, num_blocks=num_blocks)


def make_norev_variant(base_ch: int, num_blocks: int) -> nn.Module:
    return ADMUNet(in_ch=3, base_ch=base_ch, num_blocks=num_blocks, use_checkpoint=False)


def make_nocol_variant(base_ch: int, num_blocks: int) -> nn.Module:
    # K will equal C and adapters act as identity
    class NoColRevUNet(RevColUNet):
        def __init__(self, in_ch=3, base_ch=base_ch, num_blocks=num_blocks):
            super().__init__(in_ch=in_ch, base_ch=base_ch, K=base_ch, num_blocks=num_blocks)
            for m in self.modules():
                if isinstance(m, RevDualStream):
                    C = m.col.C
                    K = C
                    m.col = ColumnAdapter(C, K)
                    with torch.no_grad():
                        eye = torch.eye(C).view(C, C, 1, 1)
                        m.col.proj.weight.copy_(eye)
                        m.col.recon.weight.copy_(eye)
    from train import RevDualStream, ColumnAdapter  # import inside to avoid circular at top
    return NoColRevUNet()


def run_E3_ablation(image_size: int, batch_size: int, steps: int, warmup: int, device: torch.device) -> List[Dict[str, Any]]:
    print("[E3] Starting ablation benchmarks...")
    variants = [
        ("RevCol K=8", make_rev_variant(base_ch=96, K=8, num_blocks=2)),
        ("RevCol K=16", make_rev_variant(base_ch=96, K=16, num_blocks=2)),
        ("RevCol K=32", make_rev_variant(base_ch=96, K=32, num_blocks=2)),
        ("NoRev", make_norev_variant(base_ch=96, num_blocks=2)),
        ("NoCol", make_nocol_variant(base_ch=96, num_blocks=2)),
    ]
    rows = []
    for name, model in variants:
        try:
            print(f"[E3] Variant: {name}")
            stats = benchmark_model(model, BenchConfig(image_size=image_size, batch_size=batch_size, steps=steps, warmup=warmup, base_ch=96), device)
            row = {"variant": name, **stats}
            rows.append(row)
            print(f"[E3] -> {row}")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[E3] OOM: {name}")
                rows.append({"variant": name, "peak_mem_mb": None, "img_per_sec": 0.0, "sec_per_step": None, "final_loss": None, "status": "OOM"})
            else:
                raise

    # Save and plot
    images_dir = get_images_dir()
    with open(os.path.join(images_dir, 'E3_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(rows)
    dfp = df[df['peak_mem_mb'].notnull()].copy()
    if len(dfp) == 0:
        print("[E3] No valid results to plot.")
        return rows

    plt.figure(figsize=(6, 4))
    sns.barplot(data=dfp, x='variant', y='peak_mem_mb', color='steelblue')
    plt.title('E3 Peak GPU memory by variant')
    plt.xlabel('Variant')
    plt.ylabel('Peak GPU memory (MiB)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "E3_peak_memory_ablation.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=dfp, x='variant', y='img_per_sec', color='seagreen')
    plt.title('E3 Throughput by variant')
    plt.xlabel('Variant')
    plt.ylabel('Images per second')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "E3_throughput_ablation.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=dfp, x='variant', y='sec_per_step', color='indianred')
    plt.title('E3 Step time by variant')
    plt.xlabel('Variant')
    plt.ylabel('Seconds per step (avg)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "E3_step_time_ablation.pdf"), bbox_inches="tight")
    plt.close()

    return rows


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='RevCol-Diff evaluation script')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--e1_sizes', type=int, nargs='*', default=[32, 64])
    p.add_argument('--e1_batches', type=int, nargs='*', default=[4, 8])
    p.add_argument('--run_e1', action='store_true')
    p.add_argument('--run_e3', action='store_true')
    p.add_argument('--e3_image_size', type=int, default=64)
    p.add_argument('--e3_batch_size', type=int, default=6)
    p.add_argument('--e3_steps', type=int, default=10)
    p.add_argument('--e3_warmup', type=int, default=3)
    p.add_argument('--all', action='store_true', help='Run all evaluation experiments')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"[Eval] Using device: {device}")

    if args.all or args.run_e1:
        e1_results = run_E1_memory_tradeoff(args.e1_sizes, args.e1_batches, device)
        plot_E1(e1_results)
        print("[Eval] E1 plots and results saved.")

    if args.all or args.run_e3:
        _ = run_E3_ablation(args.e3_image_size, args.e3_batch_size, args.e3_steps, args.e3_warmup, device)
        print("[Eval] E3 plots and results saved.")


if __name__ == '__main__':
    main()
