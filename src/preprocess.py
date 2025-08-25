import os
import json
from typing import Tuple

import torch

# Headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(style="whitegrid", font_scale=1.1)

# Paths

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_images_dir() -> str:
    d = os.path.join(".research", "iteration1", "images")
    ensure_dir(d)
    return d


def get_data_dir() -> str:
    d = os.path.join("data")
    ensure_dir(d)
    return d


# Synthetic preview generator (uses train.py generators via import)
from train import gen_checker, gen_stripes, gen_gaussian_blobs, gen_radial_gradient


def save_image_grid(t: torch.Tensor, path: str, nrow: int = 4):
    # t: BxCxHxW in [-1,1]
    B, C, H, W = t.shape
    nrow = min(nrow, B)
    ncol = (B + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i in range(ncol * nrow):
        ax = axes[i]
        ax.axis('off')
        if i < B:
            img = (t[i].permute(1, 2, 0).clamp(-1, 1) + 1) / 2.0
            ax.imshow(img.cpu().numpy())
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def prepare_synthetic_preview(image_size: int = 32, batch: int = 8):
    images_dir = get_images_dir()
    x1 = gen_checker(batch, image_size)
    x2 = gen_stripes(batch, image_size, 'h')
    x3 = gen_gaussian_blobs(batch, image_size)
    x4 = gen_radial_gradient(batch, image_size)

    save_image_grid(x1, os.path.join(images_dir, f"synthetic_checker_{image_size}.pdf"))
    save_image_grid(x2, os.path.join(images_dir, f"synthetic_stripes_{image_size}.pdf"))
    save_image_grid(x3, os.path.join(images_dir, f"synthetic_blobs_{image_size}.pdf"))
    save_image_grid(x4, os.path.join(images_dir, f"synthetic_radial_{image_size}.pdf"))

    meta = {
        'image_size': image_size,
        'batch_preview': batch,
        'files': [
            f"synthetic_checker_{image_size}.pdf",
            f"synthetic_stripes_{image_size}.pdf",
            f"synthetic_blobs_{image_size}.pdf",
            f"synthetic_radial_{image_size}.pdf",
        ]
    }
    with open(os.path.join(images_dir, 'preprocess_synthetic_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Preprocess] Synthetic previews saved to {images_dir}")


def prepare_cifar10(download: bool = True):
    # optional: download CIFAR-10 for real-data experiments
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        print("[Preprocess] torchvision not available, skipping CIFAR-10 download.")
        return

    data_dir = os.path.join(get_data_dir(), 'cifar10')
    ensure_dir(data_dir)
    tfm = transforms.ToTensor()
    _ = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=tfm)
    _ = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=tfm)
    print(f"[Preprocess] CIFAR-10 prepared under {data_dir}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Preprocess script')
    p.add_argument('--synthetic_only', action='store_true', help='Only generate synthetic previews (default)')
    p.add_argument('--cifar10', action='store_true', help='Also prepare CIFAR-10 dataset')
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--batch', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    prepare_synthetic_preview(args.image_size, args.batch)
    if args.cifar10:
        prepare_cifar10(download=True)


if __name__ == '__main__':
    main()
