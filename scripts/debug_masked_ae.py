#!/usr/bin/env python3
"""Debug helper for masked AE estimator.

Reproduces the lambda sweep from `masked_ae_estimate`, saves the
kneedle input curve (log10(lambda) vs active latents and recon errors)
and, for the selected breakpoint lambda, plots the sorted absolute mask
values with the threshold line. Outputs are written under `results/debug/`.

Usage (example):
  PYTHONPATH=. python scripts/debug_masked_ae.py --manifold sphere --d 5 --n 300
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from src.data.generators import sample_sphere, sample_torus, embed_via_random_orthonormal, add_orthogonal_noise
from src.estimators import masked_ae as mdae

import torch
from torch.utils.data import DataLoader, TensorDataset


def run_lambda_sweep(X, nlatent=64, nhidden=256, lambdas=None,
                     lr=1e-3, epochs=10, batch_size=128, threshold=1e-3,
                     pretrain_epochs=50, pretrain_lr=1e-4,
                     sweep_epochs=25, sweep_lr=1e-5,
                     device=None, outdir='results/debug'):
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = np.asarray(X)
    n, D = X.shape
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if lambdas is None:
        lambdas = list(np.logspace(np.log10(0.1), np.log10(10.0), 10))

    lams = []
    acts = []
    recons = []

    # Pretrain (warm start) without sparsity
    base_model = mdae.AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
    opt_pre = torch.optim.Adam(base_model.parameters(), lr=pretrain_lr)
    for ep in range(pretrain_epochs):
        base_model.train()
        for (batch,) in dl:
            batch = batch.to(device)
            opt_pre.zero_grad()
            x_hat, z = base_model(batch)
            recon = torch.nn.functional.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
            loss = recon
            loss.backward()
            opt_pre.step()

    pretrained_state = base_model.state_dict()

    # Sweep lambdas warm-starting from pretrained weights
    for lam in lambdas:
        model = mdae.AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
        model.load_state_dict(pretrained_state)
        opt = torch.optim.Adam(model.parameters(), lr=sweep_lr)

        for ep in range(sweep_epochs):
            model.train()
            for (batch,) in dl:
                batch = batch.to(device)
                opt.zero_grad()
                x_hat, z = model(batch)
                recon = torch.nn.functional.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                # penalize masked activations (mean L1 over batch and latents)
                spars = torch.mean(torch.abs(z * model.w))
                loss = recon + lam * spars
                loss.backward()
                opt.step()

        # eval
        model.eval()
        total_recon = 0.0
        with torch.no_grad():
            for (batch,) in dl:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                recon = torch.nn.functional.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                total_recon += float(recon.item())
        total_recon /= float(n)

        w = model.w.detach().cpu().abs().numpy()
        active = int((w > threshold).sum())

        lams.append(lam)
        recons.append(total_recon)
        acts.append(active)

    lams = np.asarray(lams, dtype=float)
    recons = np.asarray(recons, dtype=float)
    acts = np.asarray(acts, dtype=int)

    # save arrays
    ts = int(time.time())
    np.savez(os.path.join(outdir, f'masked_ae_sweep_{ts}.npz'), lams=lams, acts=acts, recons=recons)

    # plot kneedle input: acts vs log10(lambda) and recon on secondary axis
    fig, ax1 = plt.subplots()
    log_lams = np.log10(lams)
    ax1.plot(log_lams, acts, marker='o', label='active latents', color='C0')
    ax1.set_xlabel('log10(lambda)')
    ax1.set_ylabel('active latents', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    ax2 = ax1.twinx()
    ax2.plot(log_lams, recons, marker='x', label='recon error', color='C1')
    ax2.set_ylabel('recon error', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    plt.title('masked-AE sweep: active latents and recon')
    plt.tight_layout()
    sweep_png = os.path.join(outdir, f'masked_ae_kneedle_input_{ts}.png')
    plt.savefig(sweep_png)
    plt.close(fig)

    # run kneedle on log_lams -> acts
    # determine breakpoint via piecewise fit on MSE vs lambda
    try:
        order = np.argsort(lams)
        x_s = lams[order]
        y_s = recons[order]

        # enforce monotone increasing recon curve (isotonic) to reduce wiggles
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=True)
            y_iso_s = ir.fit_transform(x_s, y_s)
        except Exception:
            y_iso_s = np.maximum.accumulate(y_s)

        bp = mdae._piecewise_breakpoint(x_s, y_iso_s)
        lam_bp = float(bp)
    except Exception:
        lam_bp = float(lams[len(lams) // 2])

    # draw vertical line at detected breakpoint on the plot (lambda on x-axis)
    try:
        fig, ax1 = plt.subplots()
        ax1.plot(lams, acts, marker='o', label='active latents', color='C0')
        ax1.set_xscale('log')
        ax1.set_xlabel('lambda')
        ax1.set_ylabel('active latents', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.axvline(lam_bp, color='k', linestyle='--', label=f'break@{lam_bp:.3g}')

        ax2 = ax1.twinx()
        ax2.plot(lams, recons, marker='x', label='recon error', color='C1')
        ax2.set_xscale('log')
        ax2.set_ylabel('recon error', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        plt.title('masked-AE sweep: active latents and recon (with breakpoint)')
        ax1.legend(loc='upper left')
        plt.tight_layout()
        sweep_png = os.path.join(outdir, f'masked_ae_kneedle_input_{ts}.png')
        plt.savefig(sweep_png)
        plt.close(fig)
    except Exception:
        pass

    # print the detected breakpoint
    print(f"Detected breakpoint: lam_bp={lam_bp:.6g}")

    # retrain at lam_bp to get final w
    model = mdae.AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat, z = model(batch)
            recon = torch.nn.functional.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
            spars = torch.mean(torch.abs(z * model.w))
            loss = recon + lam_bp * spars
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        w_final = model.w.detach().cpu().abs().numpy()

    # use Kneedle on -abs(w_sorted) vs sorted index to pick final estimate
    w_sorted = np.sort(w_final)[::-1]
    idxs = np.arange(len(w_sorted), dtype=float)
    ys_k = -w_sorted  # as specified: -abs(w) vs sorted index

    def _kneedle(xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.size == 0:
            return float('nan')
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]
        if np.unique(xs_s).size < 2:
            return float(xs_s[0])

        def _norm(a):
            a_min = a.min()
            a_max = a.max()
            if a_max <= a_min:
                return np.zeros_like(a)
            return (a - a_min) / (a_max - a_min)

        x_n = _norm(xs_s)
        y_n = _norm(ys_s)

        decreasing = y_n[0] > y_n[-1]
        if decreasing:
            score = x_n - y_n
        else:
            score = y_n - x_n

        if score.size >= 3:
            kernel = np.ones(3) / 3.0
            score_smooth = np.convolve(score, kernel, mode='same')
        else:
            score_smooth = score

        idx = int(np.argmax(score_smooth))
        return float(xs_s[idx])

    try:
        bp_idx = _kneedle(idxs, ys_k)
        # map bp_idx (x-coordinate) to integer count (1-based)
        est_count = int(np.round(bp_idx)) + 1
    except Exception:
        est_count = int(len(w_sorted) // 2)

    # plot sorted w with vertical line at selected index
    try:
        fig2, ax = plt.subplots()
        ax.plot(idxs + 1, w_sorted, marker='o')
        ax.axvline(bp_idx + 1, color='k', linestyle='--', label=f'knee@{int(bp_idx)+1}')
        ax.set_xlabel('latent index (sorted)')
        ax.set_ylabel('abs(w)')
        ax.set_title(f'sorted mask values at lam={lam_bp:.4g} (est_active={est_count})')
        ax.legend()
        plt.tight_layout()
        mask_png = os.path.join(outdir, f'masked_ae_sorted_mask_{ts}.png')
        plt.savefig(mask_png)
        plt.close(fig2)
    except Exception:
        mask_png = os.path.join(outdir, f'masked_ae_sorted_mask_{ts}.png')

    # save final mask values and metadata
    try:
        bp_log = float(np.log10(lam_bp)) if lam_bp > 0 else float('nan')
    except Exception:
        bp_log = float('nan')
    meta = {
        'lam_bp': float(lam_bp),
        'bp_log': bp_log,
        'est_active': int(est_count),
        'threshold': float(threshold),
        'nlatent': int(nlatent),
        'nhidden': int(nhidden),
    }
    np.savez(os.path.join(outdir, f'masked_ae_result_{ts}.npz'), w_final=w_final, w_sorted=w_sorted, meta=meta)
    return {
        'sweep_png': sweep_png,
        'mask_png': mask_png,
        'npz_sweep': os.path.join(outdir, f'masked_ae_sweep_{ts}.npz'),
        'npz_result': os.path.join(outdir, f'masked_ae_result_{ts}.npz'),
        'lam_bp': lam_bp,
        'bp_log': bp_log,
        'est_active': int(est_count),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifold', choices=['sphere', 'torus'], default='sphere')
    p.add_argument('--d', type=int, default=5)
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--D', type=int, default=None)
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--nlatent', type=int, default=64)
    p.add_argument('--nhidden', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate for Adam')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--lambdas', type=str, default=None,
                   help='comma-separated list of lambda values (overrides default logspace)')
    p.add_argument('--pretrain-epochs', type=int, default=50)
    p.add_argument('--pretrain-lr', type=float, default=1e-4)
    p.add_argument('--sweep-epochs', type=int, default=25)
    p.add_argument('--sweep-lr', type=float, default=1e-5)
    p.add_argument('--outdir', type=str, default='results/debug')
    args = p.parse_args()

    if args.manifold == 'sphere':
        X = sample_sphere(args.d, args.n, random_state=args.seed)
    else:
        X = sample_torus(args.d, args.n, random_state=args.seed)
    D = args.D or (args.d + 1 if args.manifold == 'sphere' else 2 * args.d)
    X = embed_via_random_orthonormal(X, D, random_state=args.seed)
    X = add_orthogonal_noise(X, args.sigma, random_state=args.seed)

    if args.lambdas:
        lambdas = [float(x) for x in args.lambdas.split(',')]
    else:
        lambdas = None

    out = run_lambda_sweep(X, nlatent=args.nlatent, nhidden=args.nhidden, lambdas=lambdas,
                           lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                           threshold=args.threshold, pretrain_epochs=args.pretrain_epochs,
                           pretrain_lr=args.pretrain_lr, sweep_epochs=args.sweep_epochs,
                           sweep_lr=args.sweep_lr, device=None, outdir=args.outdir)

    print('Wrote:', out)


if __name__ == '__main__':
    main()
