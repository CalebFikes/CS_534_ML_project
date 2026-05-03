#!/usr/bin/env python3
"""Repair SMAE entries in results CSV by re-running masked-AE on saved data.

Usage: repair_masked_ae.py --csv results/synthetic_large.task0.csv --out results/synthetic_large.task0.repaired.csv

The script will not overwrite the original CSV unless --overwrite is set.
It attempts to load existing datasets/ and data/mnist_latents files to avoid regenerating data.
"""
import argparse
import os
import glob
import logging
import numpy as np
import pandas as pd
import time

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from src.estimators.estimators import estimate
from src.data.generators import sample_sphere, sample_torus, embed_via_random_orthonormal, add_orthogonal_noise


def find_mnist_latents(data_dir, k, rep):
    pattern = os.path.join(data_dir, f'mnist_latents_k{int(k)}_r{int(rep)}*.npy')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # try without lr suffix
    p2 = os.path.join(data_dir, f'mnist_latents_k{int(k)}_r{int(rep)}.npy')
    if os.path.exists(p2):
        return p2
    return None


def find_synthetic_dataset(dsets_dir, d, sigma):
    # try several filename encodings used in this project
    candidates = []
    sigma_s = str(sigma)
    # common formats
    candidates.append(f'b{int(d)}_n{int(sigma)}.npy')
    candidates.append(f'b{int(d)}_n{sigma_s}.npy')
    candidates.append(f'b{int(d)}_n{sigma_s.replace(".","p")}.npy')
    candidates.append(f'b{int(d)}_n{int(round(sigma*100))}.npy')
    for c in candidates:
        p = os.path.join(dsets_dir, c)
        if os.path.exists(p):
            return p
    # fallback: glob any b{d}_n* file
    globp = os.path.join(dsets_dir, f'b{int(d)}_n*.npy')
    matches = glob.glob(globp)
    return matches[0] if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=False)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--datasets-dir', default='datasets')
    p.add_argument('--base-seed', type=int, default=0,
                   help='base_seed used when saving mnist_latents (default: 0)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-samples', type=int, default=5000,
                   help='If dataset has more than this many samples, subsample for repair (speeds up repair)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('repair_masked_ae')

    # report device availability
    cuda_avail = False
    try:
        if TORCH_AVAILABLE:
            cuda_avail = bool(torch.cuda.is_available())
    except Exception:
        cuda_avail = False
    log.info(f'torch_available={TORCH_AVAILABLE} cuda_available={cuda_avail}')

    df = pd.read_csv(args.csv)
    out_csv = args.out if args.out else (args.csv + '.repaired.csv')

    # determine csv type by filename prefix so we only use on-disk saved
    # artifacts for MNIST CSVs. For synthetic CSVs regenerate X deterministically
    # from (manifold,d,n,seed) to ensure exact reproducibility.
    csv_base = os.path.basename(args.csv).lower()
    is_mnist_csv = csv_base.startswith('mnist')
    is_synthetic_csv = csv_base.startswith('synthetic')

    mask_kwargs = {
        'lr': 5e-5,
        'pretrain_epochs': 10,
        'pretrain_lr': 5e-5,
        'sweep_epochs': 25,
        'sweep_lr': 1e-5,
        # retrain epochs (used as `epochs` by masked_ae_estimate)
        'epochs': 35,
    }

    # Find rows needing repair and group them by parameters that matter for SMAE
    mask_rows = []
    for idx, row in df.iterrows():
        if str(row.get('estimator', '')).strip().upper() not in ('SMAE', 'MASKED-AE'):
            continue
        try:
            est = float(row.get('estimate', np.nan))
            if np.isnan(est):
                mask_rows.append((idx, row))
        except Exception:
            mask_rows.append((idx, row))

    if not mask_rows:
        log.info('No SMAE rows needing repair found.')
    repaired = 0

    # Build grouping key: for MNIST use (bottleneck, sigma, seed), for synthetic use (manifold,d,n,sigma,seed)
    def make_key(row):
        if is_mnist_csv:
            b = int(row.get('bottleneck', -1))
            s = float(row.get('sigma', 0.0))
            seed = int(row.get('seed', 0))
            return ('mnist', b, s, seed)
        else:
            manifold = str(row.get('manifold', 'sphere')).lower()
            d = int(row.get('d', -1))
            s = float(row.get('sigma', 0.0))
            seed = int(row.get('seed', 0))
            # Group by manifold, d, sigma, seed (ignore `n` since we always use 60000 samples)
            return ('synth', manifold, d, s, seed)

    # Group indices by key
    groups = {}
    for idx, row in mask_rows:
        key = make_key(row)
        groups.setdefault(key, []).append((idx, row))

    # Process each group once and replicate the result across rows differing only by k
    for key, members in groups.items():
        rep_idx, rep_row = members[0]
        log.info(f'Processing group key={key} with {len(members)} rows; example row {rep_idx}')

        # Build X for representative row
        X = None
        if key[0] == 'mnist':
            b, sigma, seed = key[1], key[2], key[3]
            rep = int(seed - int(args.base_seed))
            lat = find_mnist_latents(args.data_dir, b, rep)
            if lat is None:
                log.warning(f'No mnist latents found for bottleneck={b} rep={rep} (seed={seed})')
                for idx, _ in members:
                    df.at[idx, 'estimate'] = np.nan
                    df.at[idx, 'error'] = f'No mnist latents for b={b} rep={rep}'
                continue
            Z = np.load(lat)
            rng = np.random.RandomState(int(seed))
            X = Z + rng.normal(scale=float(sigma), size=Z.shape)
        else:
            _, manifold, d, sigma, seed = key
            # always use full dataset size (60000) when regenerating synthetic data
            n = 60000
            if int(d) < 0 or int(n) <= 0:
                log.warning(f'Insufficient synthetic metadata for group {key}')
                for idx, _ in members:
                    df.at[idx, 'estimate'] = np.nan
                    df.at[idx, 'error'] = f'Insufficient metadata d={d} n={n}'
                continue
            log.info(f'Regenerating synthetic X for group manifold={manifold} d={d} n={n} seed={seed} sigma={sigma}')
            try:
                if manifold == 'sphere':
                    Xgen = sample_sphere(int(d), int(n), random_state=int(seed))
                else:
                    Xgen = sample_torus(int(d), int(n), random_state=int(seed))
                D = int(d) + 1 if manifold == 'sphere' else 2 * int(d)
                Xemb = embed_via_random_orthonormal(Xgen, D, random_state=int(seed))
                X = add_orthogonal_noise(Xemb, float(sigma), random_state=int(seed))
            except Exception as e:
                log.warning(f'Failed to regenerate synthetic X for group {key}: {e}')
                for idx, _ in members:
                    df.at[idx, 'estimate'] = np.nan
                    df.at[idx, 'error'] = f'Regen failed: {e}'
                continue

        if X is None:
            continue

        # Subsample if requested
        try:
            max_s = int(args.max_samples) if args.max_samples is not None else None
        except Exception:
            max_s = None
        if max_s and hasattr(X, 'shape') and X.shape[0] > max_s:
            sample_seed = int(rep_row.get('seed', 0)) if 'seed' in rep_row else 0
            rng = np.random.RandomState(sample_seed)
            idxs = rng.choice(X.shape[0], size=max_s, replace=False)
            log.info(f'Subsampling dataset for group {key} from {X.shape[0]} -> {max_s} using seed={sample_seed}')
            X = X[idxs]

        if args.dry_run:
            log.info('dry-run: would run masked-ae on loaded X for group %s', key)
            continue

        # Run estimate once for this group
        try:
            device = None
            if TORCH_AVAILABLE:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            t0 = time.time()
            val = estimate(X, method='masked-ae', return_debug=False, device=device, **mask_kwargs)
            elapsed = time.time() - t0
            for idx, _ in members:
                df.at[idx, 'estimate'] = float(val)
                df.at[idx, 'error'] = ''
            repaired += len(members)
            log.info(f'Group {key} repaired: estimate={val} elapsed_s={elapsed:.2f} device={device} samples={getattr(X,"shape",(None,))[0]} rows={len(members)}')
        except Exception as e:
            for idx, _ in members:
                df.at[idx, 'estimate'] = np.nan
                df.at[idx, 'error'] = str(e).replace('\n', ' | ')
            log.exception(f'Failed to run masked-ae for group {key}')

    if repaired == 0:
        log.info('No rows repaired.')

    if args.overwrite:
        final_out = args.csv
    else:
        final_out = out_csv

    if not args.dry_run:
        df.to_csv(final_out, index=False)
        log.info(f'Wrote repaired CSV to {final_out}')


if __name__ == '__main__':
    main()
