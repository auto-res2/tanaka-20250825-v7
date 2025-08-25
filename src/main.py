import os
import json
import yaml

from typing import Any, Dict

from train import TrainConfig, train_one, set_seed, get_device
from evaluate import run_E1_memory_tradeoff, plot_E1, run_E3_ablation
from preprocess import prepare_synthetic_preview, prepare_cifar10, get_images_dir


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_quick_test():
    print("==== Quick functional test: starting ====")
    set_seed(123)
    device = get_device()
    print(f"Using device: {device}")

    # Preprocess: synthetic previews
    prepare_synthetic_preview(image_size=32, batch=8)

    # Train small models
    cfg_rev = TrainConfig(model='revcol', image_size=32, batch_size=8, steps=60, lr=1e-3, base_ch=64, K=16)
    stats_rev = train_one(cfg_rev, device)

    cfg_adm = TrainConfig(model='adm', image_size=32, batch_size=8, steps=60, lr=1e-3, base_ch=64)
    stats_adm = train_one(cfg_adm, device)

    # Evaluate E1 small grid
    e1_results = run_E1_memory_tradeoff(sizes=[32, 64], batches=[4, 8], device=device)
    plot_E1(e1_results)

    # Evaluate E3 small ablation
    _ = run_E3_ablation(image_size=64, batch_size=6, steps=10, warmup=3, device=device)

    # Save quick summary
    images_dir = get_images_dir()
    summary = {
        'train_revcol': {'final_val_loss': stats_rev['val_mean']},
        'train_adm': {'final_val_loss': stats_adm['val_mean']},
    }
    with open(os.path.join(images_dir, 'quick_test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("==== Quick functional test: completed. Figures saved under .research/iteration1/images ====")


def main():
    import argparse
    p = argparse.ArgumentParser(description='RevCol-Diff end-to-end experiment runner')
    p.add_argument('--config', type=str, default='config/config.yaml')
    p.add_argument('--quick-test', action='store_true', help='Run a fast, self-contained test')
    p.add_argument('--prepare-cifar10', action='store_true', help='Also download CIFAR-10 during preprocess')
    args = p.parse_args()

    if args.quick_test:
        run_quick_test()
        return

    cfg = load_config(args.config)

    # Preprocess
    ims = cfg.get('preprocess', {}).get('synthetic_preview', {})
    prepare_synthetic_preview(image_size=int(ims.get('image_size', 32)), batch=int(ims.get('batch', 8)))
    if args.prepare_cifar10 or cfg.get('preprocess', {}).get('prepare_cifar10', False):
        prepare_cifar10(download=True)

    # Train (can train multiple models)
    device = get_device()
    train_jobs = cfg.get('train', [])
    for job in train_jobs:
        tc = TrainConfig(
            model=job.get('model', 'revcol'),
            image_size=int(job.get('image_size', 32)),
            batch_size=int(job.get('batch_size', 8)),
            steps=int(job.get('steps', 200)),
            lr=float(job.get('lr', 1e-3)),
            base_ch=int(job.get('base_ch', 64)),
            K=int(job.get('K', 16)),
            num_blocks=int(job.get('num_blocks', 2)),
            patterns_train=tuple(job.get('patterns_train', ["checker", "blobs", "stripes_h", "radial"])),
            patterns_val=tuple(job.get('patterns_val', ["checker", "stripes_v", "blobs", "noise"]))
        )
        print(f"[Main] Training job: {tc}")
        _ = train_one(tc, device)

    # Evaluate
    eval_cfg = cfg.get('evaluate', {})
    if eval_cfg.get('run_e1', True):
        e1_sizes = eval_cfg.get('e1_sizes', [32, 64])
        e1_batches = eval_cfg.get('e1_batches', [4, 8])
        e1_results = run_E1_memory_tradeoff(e1_sizes, e1_batches, device)
        plot_E1(e1_results)
    if eval_cfg.get('run_e3', True):
        e3 = eval_cfg.get('e3', {})
        _ = run_E3_ablation(
            image_size=int(e3.get('image_size', 64)),
            batch_size=int(e3.get('batch_size', 6)),
            steps=int(e3.get('steps', 10)),
            warmup=int(e3.get('warmup', 3)),
            device=device
        )


if __name__ == '__main__':
    main()
