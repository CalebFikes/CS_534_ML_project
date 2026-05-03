#!/usr/bin/env python3
"""Produce two figures:
A) MNIST reconstructions across bottleneck sizes
B) Masked-AE lambda sweep and mask-knee diagnostics

Runs a quick smoke test by default (subset) unless `--full` is passed.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.estimators.masked_ae import masked_ae_estimate


class AE(nn.Module):
    def __init__(self, nambient, nlatent=64, nhidden=400):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nambient, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nlatent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nlatent, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nambient),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #send x to GPU
        x = x.to(next(self.parameters()).device)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def load_mnist(data_path):
    f = np.load(data_path)
    keys = f.files
    if 'images' in keys:
        X = f['images']
    elif 'x' in keys:
        X = f['x']
    else:
        X = f[keys[0]]
    # ensure shape (N, 28, 28) or (N, 784)
    if X.ndim == 3:
        N, H, W = X.shape
        X = X.reshape(N, H * W)
    
    if X.max() > 1.0:
        X = X / 255.0
    return X.astype(np.float32)


def train_ae(X, bottleneck, epochs=100, lr=5e-5, batch_size=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n, D = X.shape
    model = AE(nambient=D, nlatent=bottleneck, nhidden=400).to(device)
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat = model(batch)
            loss = loss_fn(x_hat, batch.view(batch.size(0), -1))
            loss.backward()
            opt.step()
    return model


def fig_a(X, bottlenecks, epochs, lr, subset, outdir):
    os.makedirs(outdir, exist_ok=True)
    if subset is not None and subset > 0:
        X = X[:subset]
    # pick last image
    last_img = X[-1]
    imgs = [last_img.reshape(28, 28)]
    labels = ['true']
    for k in bottlenecks:
        model = train_ae(X, bottleneck=k, epochs=epochs, lr=lr)
        model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(last_img).float().unsqueeze(0)
            recon = model(inp).cpu().numpy().reshape(-1)
        imgs.append(recon.reshape(28, 28))
        labels.append(f'bottleneck = {k}')

    # plot row
    fig, axes = plt.subplots(1, len(imgs), figsize=(1.8 * len(imgs), 2.2))
    for ax, im, lab in zip(axes, imgs, labels):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(lab)
    fig.suptitle('MNIST reconstructions across bottleneck sizes')
    out = os.path.join(outdir, 'mnist_ae_compression.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def fig_b(X, outdir, subset, **mask_kwargs):
    os.makedirs(outdir, exist_ok=True)
    if subset is not None and subset > 0:
        X = X[:subset]
    debug = masked_ae_estimate(X, return_debug=True, **mask_kwargs)
    lams = debug.get('lams') if debug.get('lams') is not None else debug.get('lambdas')
    recons = debug.get('recons') if debug.get('recons') is not None else debug.get('recons')
    lam_bp = debug.get('lam_bp')
    w_sorted = np.sort(debug.get('w_final', np.array([])))[::-1]
    est_active = int(debug.get('meta', {}).get('est_active', len(w_sorted)//2))

    # two vertically aligned subplots: top = MSE vs lambda, bottom = abs(w) vs ordered index
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharex=False)
    ax_top, ax_bot = axes

    # Top: MSE vs lambda (log-x)
    ax_top.plot(lams, recons, '-o', color='C0')
    ax_top.set_xscale('log')
    ax_top.set_xlabel('lambda')
    ax_top.set_ylabel('reconstruction MSE')
    ax_top.grid(True, which='both', ls='--', alpha=0.4)
    if lam_bp is not None:
        ax_top.axvline(lam_bp, color='C1', linestyle='--', label=f'lam_bp={lam_bp:.3g}')
        ax_top.legend()

    # Bottom: abs(w) vs sorted index (descending)
    indices = np.arange(1, len(w_sorted) + 1)
    ax_bot.plot(indices, np.abs(w_sorted), marker='o', color='C2')
    ax_bot.set_xlabel('sorted index (descending)')
    ax_bot.set_ylabel('abs(w)')
    ax_bot.grid(True, ls='--', alpha=0.4)
    # vertical line at estimated active count
    if est_active is not None and est_active > 0:
        ax_bot.axvline(est_active, color='C1', linestyle='--', label=f'est_active={est_active}')
        ax_bot.legend()

    fig.suptitle('masked-AE sweep and mask knee')
    out = os.path.join(outdir, 'masked_ae_sweep_and_knee.png')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/mnist.npz')
    p.add_argument('--outdir', default='results/figs_large')
    p.add_argument('--bottlenecks', default='5,7,9,12,15')
    p.add_argument('--ae-epochs', type=int, default=100)
    p.add_argument('--ae-lr', type=float, default=5e-5)
    p.add_argument('--subset', type=int, default=2000)
    p.add_argument('--run-masked', action='store_true')
    args = p.parse_args()

    X = load_mnist(args.data)
    bks = [int(x) for x in args.bottlenecks.split(',') if x.strip()]
    fig_a(X, bks, epochs=args.ae_epochs, lr=args.ae_lr, subset=args.subset, outdir=args.outdir)
    if args.run_masked:
        mask_kwargs = {
            'nlatent': 64,
            'nhidden': 256,
            'pretrain_epochs': 10,
            'pretrain_lr': 5e-5,
            'sweep_epochs': 25,
            'sweep_lr': 1e-5,
            'lr': 5e-5,
        }
        fig_b(X, args.outdir, subset=args.subset, **mask_kwargs)


if __name__ == '__main__':
    main()
