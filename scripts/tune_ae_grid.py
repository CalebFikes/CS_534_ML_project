#!/usr/bin/env python3
import argparse
import subprocess
import shlex
import os
import numpy as np
import shutil


def run_train(cmd):
    print(f"[TUNE] Running: {cmd}", flush=True)
    res = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = res.stdout.decode('utf-8', errors='ignore')
    print(out, flush=True)
    return res.returncode, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bottleneck', type=int, required=True)
    parser.add_argument('--r', type=int, required=True)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr-grid', default='1e-3,5e-4,1e-4')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--out-model-template', default='models/ae_k{K}_r{R}_lr{LR}.pth')
    parser.add_argument('--out-latent-template', default='data/mnist_latents_k{K}_r{R}_lr{LR}.npy')
    parser.add_argument('--out-loss-template', default='models/ae_k{K}_r{R}_lr{LR}_loss.npy')
    parser.add_argument('--final-model', default='models/ae_k{K}_r{R}.pth')
    parser.add_argument('--final-latents', default='data/mnist_latents_k{K}_r{R}.npy')
    args = parser.parse_args()

    lrs = [float(x) for x in args.lr_grid.split(',') if x]
    seeds = [int(x) for x in args.seeds.split(',') if x]

    results = []
    for lr in lrs:
        for seed in seeds:
            model_path = args.out_model_template.format(K=args.bottleneck, R=args.r, LR=str(lr).replace('.','p'))
            latent_path = args.out_latent_template.format(K=args.bottleneck, R=args.r, LR=str(lr).replace('.','p'))
            loss_path = args.out_loss_template.format(K=args.bottleneck, R=args.r, LR=str(lr).replace('.','p'))
            os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
            cmd = (
                f"python -m src.models.train_autoencoder --data-dir {args.data_dir} --bottleneck {args.bottleneck} "
                f"--epochs {args.epochs} --batch-size {args.batch_size} --hidden-dim {args.hidden_dim} "
                f"--num-workers {args.num_workers} --seed {seed} --lr {lr} "
                f"--save-model {model_path} --save-latents {latent_path} --save-loss {loss_path}"
            )
            rc, out = run_train(cmd)
            final_loss = None
            if os.path.exists(loss_path):
                try:
                    arr = np.load(loss_path)
                    if arr.size > 0:
                        final_loss = float(arr[-1])
                except Exception:
                    final_loss = None
            results.append({'lr': lr, 'seed': seed, 'model': model_path, 'latents': latent_path, 'loss': final_loss})

    # pick best by lowest final loss (ignore None)
    valid = [r for r in results if r['loss'] is not None]
    if not valid:
        print('[TUNE] No valid runs produced losses; aborting', flush=True)
        return 2
    best = min(valid, key=lambda x: x['loss'])
    print(f"[TUNE] Best LR={best['lr']} seed={best['seed']} loss={best['loss']}", flush=True)

    final_model = args.final_model.format(K=args.bottleneck, R=args.r)
    final_latent = args.final_latents.format(K=args.bottleneck, R=args.r)
    os.makedirs(os.path.dirname(final_model) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(final_latent) or '.', exist_ok=True)
    shutil.copy(best['model'], final_model)
    shutil.copy(best['latents'], final_latent)
    print(f"[TUNE] Copied best model to {final_model} and latents to {final_latent}", flush=True)

    # print per-epoch losses of best
    best_loss_path = args.out_loss_template.format(K=args.bottleneck, R=args.r, LR=str(best['lr']).replace('.','p'))
    try:
        arr = np.load(best_loss_path)
        for i, v in enumerate(arr, 1):
            print(f"[AE EPOCH BEST] epoch={i} loss={v:.6f}", flush=True)
    except Exception as e:
        print(f"[TUNE] Could not load best loss array: {e}", flush=True)


if __name__ == '__main__':
    main()
