"""Full experiment runner.

Supports a small smoke configuration and parallelization across Monte Carlo
replicates. Writes raw estimates to CSV under `results/` and saves a couple of
diagnostic plots for the smoke test.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.data.generators import sample_sphere, sample_torus, embed_via_random_orthonormal, add_orthogonal_noise
from src.estimators.estimators import estimate
import matplotlib.pyplot as plt


def synthetic_worker(task):
    """Worker function that generates one replicate and returns list of records."""
    manifold = task['manifold']
    d = task['d']
    n = task['n']
    D = task['D']
    sigma = task['sigma']
    methods = task['methods']
    K = task['K']
    seed = task['seed']

    if manifold == 'sphere':
        X = sample_sphere(d, n, random_state=seed)
    else:
        X = sample_torus(d, n, random_state=seed)
    X = embed_via_random_orthonormal(X, D, random_state=seed)
    X = add_orthogonal_noise(X, sigma, random_state=seed)

    records = []
    for m in methods:
        for k in K:
            err_msg = ''
            try:
                val = estimate(X, method=m, k=k)
            except TypeError:
                # some estimators don't accept k; try without
                try:
                    val = estimate(X, method=m)
                except Exception:
                    val = float('nan')
                    err_msg = traceback.format_exc()
            except Exception:
                val = float('nan')
                err_msg = traceback.format_exc()
            records.append({
                'estimator': m,
                'manifold': manifold,
                'd': d,
                'sigma': sigma,
                'k': k,
                'n': n,
                'estimate': float(val),
                'seed': int(seed),
                'error': err_msg.replace('\n', ' | '),
            })
    return records


def run_synthetic(config, out_csv, max_workers=None):
    tasks = []
    methods = config['methods']
    paired = config.get('paired', False)
    for manifold in config['manifolds']:
        if not paired:
            for d in config['intrinsic_dims']:
                D = config.get('ambient_dim')
                if D is None:
                    D = d + 1 if manifold == 'sphere' else 2 * d
                for sigma in config['noise_levels']:
                    for r in range(config['R']):
                        seed = config.get('base_seed', 0) + r
                        tasks.append({
                            'manifold': manifold,
                            'd': d,
                            'n': config['n_samples'],
                            'D': D,
                            'sigma': sigma,
                            'methods': methods,
                            'K': config['neighbor_grid_K'],
                            'seed': seed,
                        })
        else:
            # paired mode: iterate zipped lists of same length (intrinsic_dims, noise_levels)
            dims = config['intrinsic_dims']
            noises = config['noise_levels']
            if len(dims) != len(noises):
                raise ValueError('paired mode requires intrinsic_dims and noise_levels of same length')
            for d, sigma in zip(dims, noises):
                D = config.get('ambient_dim')
                if D is None:
                    D = d + 1 if manifold == 'sphere' else 2 * d
                for r in range(config['R']):
                    seed = config.get('base_seed', 0) + r
                    tasks.append({
                        'manifold': manifold,
                        'd': d,
                        'n': config['n_samples'],
                        'D': D,
                        'sigma': sigma,
                        'methods': methods,
                        'K': config['neighbor_grid_K'],
                        'seed': seed,
                    })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(synthetic_worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                records.extend(rec)
            except Exception as e:
                print('Worker error', e)

    df = pd.DataFrame.from_records(records)
    # When running as a Slurm array task, avoid multiple processes overwriting
    # the same output file. If `SLURM_ARRAY_TASK_ID` is present, write a
    # per-task file (caller can aggregate later).
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_INDEX')
    if task_id:
        base, ext = os.path.splitext(out_csv)
        out_task = f"{base}.task{task_id}{ext}"
        df.to_csv(out_task, index=False)
        print(f'Wrote per-task synthetic results to {out_task}')
    else:
        df.to_csv(out_csv, index=False)
        print(f'Wrote synthetic results to {out_csv}')
    return df


def run_mnist_autoencoder(config, out_csv, run_train=True):
    """Train small autoencoders and compute estimates on latent representations.

    For the smoke test we train on a subset of MNIST using the training script
    in `src.models.train_autoencoder.py` and load the saved latents.
    """
    import subprocess
    import numpy as np

    records = []
    data_dir = config.get('data_dir', 'data')
    for k in config['bottleneck_dims']:
        for r in range(config['R']):
            seed = config.get('base_seed', 0) + r
            model_path = f'models/ae_k{k}_r{r}.pth'
            latents_path = f'data/mnist_latents_k{k}_r{r}.npy'
            os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
            os.makedirs(os.path.dirname(latents_path) or '.', exist_ok=True)
            if run_train:
                cmd = [
                    'python', '-m', 'src.models.train_autoencoder',
                    '--data-dir', data_dir,
                    '--batch-size', str(config['batch_size']),
                    '--hidden-dim', str(config.get('hidden_dim', 400)),
                    '--bottleneck', str(k),
                    '--epochs', str(config['epochs']),
                    '--save-model', model_path,
                    '--save-latents', latents_path,
                    '--subset-size', str(config.get('mnist_subset_size', 0)),
                    '--num-workers', str(config.get('num_workers', 0)),
                    '--seed', str(seed),
                ]
                if 'cpu' in config and config['cpu']:
                    cmd.append('--cpu')
                subprocess.check_call(cmd)

            Z = np.load(latents_path)
            for m in config['methods']:
                for k_n in config['neighbor_grid_K']:
                    err_msg = ''
                    try:
                        val = estimate(Z, method=m, k=k_n)
                    except TypeError:
                        try:
                            val = estimate(Z, method=m)
                        except Exception:
                            val = float('nan')
                            err_msg = traceback.format_exc()
                    except Exception:
                        val = float('nan')
                        err_msg = traceback.format_exc()
                    records.append({
                        'estimator': m,
                        'bottleneck': k,
                        'k_n': k_n,
                        'estimate': float(val),
                        'seed': int(seed),
                        'error': err_msg.replace('\n', ' | '),
                    })

    df = pd.DataFrame.from_records(records)
    # See comment above in `run_synthetic` — write per-task CSV when inside
    # an array so concurrent tasks don't collide; aggregation can be run
    # after the array completes.
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID') or os.environ.get('SLURM_ARRAY_TASK_INDEX')
    if task_id:
        base, ext = os.path.splitext(out_csv)
        out_task = f"{base}.task{task_id}{ext}"
        df.to_csv(out_task, index=False)
        print(f'Wrote per-task mnist results to {out_task}')
    else:
        df.to_csv(out_csv, index=False)
        print(f'Wrote mnist results to {out_csv}')
    return df


def make_basic_plots(df_syn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # plot mean estimate vs true d for each estimator (spheres only)
    plt.figure()
    syn = df_syn[df_syn['manifold'] == 'sphere']
    grouped = syn.groupby(['estimator', 'd'])['estimate'].agg(['mean', 'std', 'count']).reset_index()
    # compute SEM where possible
    grouped['sem'] = grouped['std'] / (grouped['count'].replace(0, 1) ** 0.5)
    estimators = grouped['estimator'].unique()
    for est in estimators:
        g = grouped[grouped['estimator'] == est]
        plt.errorbar(g['d'], g['mean'], yerr=g['sem'], label=est, capsize=3, marker='o')
    # identity line for reference (y = x)
    min_d = grouped['d'].min()
    max_d = grouped['d'].max()
    xs = np.linspace(min_d, max_d, 100)
    plt.plot(xs, xs, linestyle=':', color='k', label='y = x')
    plt.xlabel('true d')
    plt.ylabel('estimate (mean ± std)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sphere_estimates_vs_d.png'))


def make_mnist_plots(df_mnist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Violation rate: fraction of estimates > bottleneck
    df_mnist['violation'] = df_mnist['estimate'] > df_mnist['bottleneck']
    vr = df_mnist.groupby(['estimator', 'bottleneck'])['violation'].mean().reset_index()
    plt.figure()
    for est in vr['estimator'].unique():
        sub = vr[vr['estimator'] == est]
        plt.plot(sub['bottleneck'], sub['violation'], marker='o', label=est)
    plt.xlabel('bottleneck')
    plt.ylabel('violation rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mnist_violation_rates.png'))

    # Boxplot of estimates per bottleneck
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.boxplot(x='bottleneck', y='estimate', hue='estimator', data=df_mnist)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mnist_estimates_boxplot.png'))


def _estimator_color_map(estimators):
    estimators = list(estimators)
    cmap = plt.get_cmap('tab10')
    colors = {est: cmap(i % 10) for i, est in enumerate(estimators)}
    return colors


def make_full_synthetic_plots(df_syn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ests = sorted(df_syn['estimator'].unique())
    colors = _estimator_color_map(ests)

    # iterate manifolds and noise levels
    for manifold in df_syn['manifold'].unique():
        sub_m = df_syn[df_syn['manifold'] == manifold]
        for sigma in sorted(sub_m['sigma'].unique()):
            sub_ms = sub_m[sub_m['sigma'] == sigma]
            # 1. Estimate vs true d
            plt.figure()
            for est in ests:
                g = sub_ms[sub_ms['estimator'] == est].groupby('d')['estimate'].agg(['mean', 'std', 'count']).reset_index()
                if g.empty:
                    continue
                g['sem'] = g['std'] / (g['count'].replace(0, 1) ** 0.5)
                plt.errorbar(g['d'], g['mean'], yerr=g['sem'], label=est, color=colors[est], capsize=3, marker='o')
            xs = np.linspace(sub_ms['d'].min(), sub_ms['d'].max(), 100)
            plt.plot(xs, xs, linestyle=':', color='k', label='y = x')
            plt.xlabel('true d')
            plt.ylabel('estimate (mean ± std)')
            plt.title(f'estimate vs d — {manifold}, sigma={sigma}')
            plt.legend()
            plt.tight_layout()
            fname = f'synthetic_estimate_vs_d_{manifold}_sigma_{sigma}.png'
            plt.savefig(os.path.join(out_dir, fname))

            # 2. MAE vs d (mean absolute error across replicates)
            plt.figure()
            for est in ests:
                sel_est = sub_ms[sub_ms['estimator'] == est]
                if sel_est.empty:
                    continue
                mae_by_d = sel_est.groupby('d').apply(lambda g: np.mean(np.abs(g['estimate'] - g.name))).reset_index(name='mae')
                plt.plot(mae_by_d['d'], mae_by_d['mae'], marker='o', label=est, color=colors[est])
            plt.xlabel('true d')
            plt.ylabel('MAE')
            plt.title(f'MAE vs d — {manifold}, sigma={sigma}')
            plt.legend()
            plt.tight_layout()
            fname = f'synthetic_mae_vs_d_{manifold}_sigma_{sigma}.png'
            plt.savefig(os.path.join(out_dir, fname))

            # 3. Bias + Variance
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            for est in ests:
                g = sub_ms[sub_ms['estimator'] == est].groupby('d')['estimate'].agg(['mean', 'std', 'count']).reset_index()
                if g.empty:
                    continue
                bias = g['mean'] - g['d']
                plt.plot(g['d'], bias, marker='o', label=est, color=colors[est])
            plt.xlabel('true d')
            plt.ylabel('bias')
            plt.title('Bias vs d')
            plt.legend()
            plt.subplot(1, 2, 2)
            for est in ests:
                g = sub_ms[sub_ms['estimator'] == est].groupby('d')['estimate'].agg(['std', 'count']).reset_index()
                if g.empty:
                    continue
                # show standard deviation as variance measure
                plt.plot(g['d'], g['std'], marker='o', label=est, color=colors[est])
            plt.xlabel('true d')
            plt.ylabel('std')
            plt.title('Variance vs d')
            plt.tight_layout()
            fname = f'synthetic_bias_var_{manifold}_sigma_{sigma}.png'
            plt.savefig(os.path.join(out_dir, fname))

    # 4. Neighborhood sensitivity (k) — fix manifold, d, sigma; iterate representative ds present
    ks = sorted(df_syn['k'].unique()) if 'k' in df_syn.columns else []
    for manifold in df_syn['manifold'].unique():
        for d in sorted(df_syn['d'].unique()):
            for sigma in sorted(df_syn['sigma'].unique()):
                sel = df_syn[(df_syn['manifold'] == manifold) & (df_syn['d'] == d) & (df_syn['sigma'] == sigma)]
                if sel.empty or ks == []:
                    continue
                plt.figure()
                for est in ests:
                    tmp = sel[sel['estimator'] == est].groupby('k')['estimate'].agg(['mean', 'std', 'count']).reset_index()
                    if tmp.empty:
                        continue
                    tmp['sem'] = tmp['std'] / (tmp['count'].replace(0, 1) ** 0.5)
                    plt.errorbar(tmp['k'], tmp['mean'], yerr=tmp['sem'], marker='o', label=est, color=colors[est], capsize=3)
                plt.xlabel('k')
                plt.ylabel('estimate')
                plt.title(f'k sensitivity — {manifold}, d={d}, sigma={sigma}')
                plt.legend()
                plt.tight_layout()
                fname = f'synthetic_k_sensitivity_{manifold}_d_{d}_sigma_{sigma}.png'
                plt.savefig(os.path.join(out_dir, fname))

    # 5. Stability summary: range across k
    for manifold in df_syn['manifold'].unique():
        for sigma in sorted(df_syn['sigma'].unique()):
            sel = df_syn[(df_syn['manifold'] == manifold) & (df_syn['sigma'] == sigma)]
            if sel.empty or 'k' not in sel.columns:
                continue
            plt.figure()
            for est in ests:
                tmp = sel[sel['estimator'] == est].groupby(['d', 'k'])['estimate'].mean().unstack()
                if tmp.empty:
                    continue
                stability = tmp.max(axis=1) - tmp.min(axis=1)
                plt.plot(stability.index, stability.values, marker='o', label=est, color=colors[est])
            plt.xlabel('true d')
            plt.ylabel('stability (max_k - min_k)')
            plt.title(f'stability vs d — {manifold}, sigma={sigma}')
            plt.legend()
            plt.tight_layout()
            fname = f'synthetic_stability_vs_d_{manifold}_sigma_{sigma}.png'
            plt.savefig(os.path.join(out_dir, fname))

    # 6. Noise robustness: compute NoiseShift relative to sigma=0
    for manifold in df_syn['manifold'].unique():
        for d in sorted(df_syn['d'].unique()):
            sel = df_syn[(df_syn['manifold'] == manifold) & (df_syn['d'] == d)]
            if sel.empty:
                continue
            # compute mean estimate per sigma per estimator
            mean_by = sel.groupby(['estimator', 'sigma'])['estimate'].mean().reset_index()
            # pivot for easy subtraction
            pivot = mean_by.pivot(index='estimator', columns='sigma', values='estimate')
            if 0 not in pivot.columns:
                # cannot compute NoiseShift without sigma=0 baseline; skip
                continue
            baseline = pivot[0]
            sigmas = sorted(mean_by['sigma'].unique())
            plt.figure()
            for est in ests:
                if est not in pivot.index:
                    continue
                shifts = [pivot.loc[est, s] - baseline.loc[est] if (s in pivot.columns and not np.isnan(pivot.loc[est, s])) else np.nan for s in sigmas]
                plt.plot(sigmas, shifts, marker='o', label=est, color=colors[est])
            plt.xlabel('sigma')
            plt.ylabel('NoiseShift (mean estimate - mean estimate at sigma=0)')
            plt.title(f'noise shift — {manifold}, d={d}')
            plt.legend()
            plt.tight_layout()
            fname = f'synthetic_noise_effect_{manifold}_d_{d}.png'
            plt.savefig(os.path.join(out_dir, fname))

    # 7. Sphere vs Torus comparison
    try:
        plt.figure()
        for est in ests:
            tmp = df_syn[df_syn['estimator'] == est].groupby(['manifold', 'd'])['estimate'].mean().unstack(level=0)
            if tmp.empty:
                continue
            # plot sphere and torus if both exist
            if 'sphere' in tmp.columns:
                plt.plot(tmp.index, tmp['sphere'], marker='o', label=f'{est} (sphere)', color=colors[est])
            if 'torus' in tmp.columns:
                plt.plot(tmp.index, tmp['torus'], marker='x', label=f'{est} (torus)', linestyle='--', color=colors[est])
        plt.xlabel('true d')
        plt.ylabel('estimate')
        plt.title('geometry comparison (sphere vs torus)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'synthetic_geometry_comparison.png'))
    except Exception:
        pass


def make_full_mnist_plots(df_mnist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if df_mnist is None or df_mnist.empty:
        return
    ests = sorted(df_mnist['estimator'].unique())
    colors = _estimator_color_map(ests)

    # 8. Estimate vs bottleneck
    plt.figure()
    for est in ests:
        tmp = df_mnist[df_mnist['estimator'] == est].groupby('bottleneck')['estimate'].agg(['mean', 'std']).reset_index()
        if tmp.empty:
            continue
        plt.errorbar(tmp['bottleneck'], tmp['mean'], yerr=tmp['std'], marker='o', label=est, color=colors[est])
    xs = sorted(df_mnist['bottleneck'].unique())
    if xs:
        plt.plot(xs, xs, linestyle=':', color='k', label='y = k')
    plt.xlabel('bottleneck')
    plt.ylabel('estimate')
    plt.title('estimate vs bottleneck')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'real_estimate_vs_bottleneck.png'))

    # 9. Violation rate (existing)
    df_mnist['violation'] = df_mnist['estimate'] > df_mnist['bottleneck']
    vr = df_mnist.groupby(['estimator', 'bottleneck'])['violation'].mean().reset_index()
    plt.figure()
    for est in vr['estimator'].unique():
        sub = vr[vr['estimator'] == est]
        plt.plot(sub['bottleneck'], sub['violation'], marker='o', label=est, color=colors[est])
    plt.xlabel('bottleneck')
    plt.ylabel('violation rate')
    plt.title('violation rate vs bottleneck')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'real_violation_rate.png'))

    # 10. Variance across runs
    plt.figure()
    for est in ests:
        tmp = df_mnist[df_mnist['estimator'] == est].groupby('bottleneck')['estimate'].agg(['var']).reset_index()
        if tmp.empty:
            continue
        plt.plot(tmp['bottleneck'], tmp['var'], marker='o', label=est, color=colors[est])
    plt.xlabel('bottleneck')
    plt.ylabel('variance')
    plt.title('variance across runs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'real_variance_vs_k.png'))

    # 11. Neighborhood sensitivity (real): use k_n
    if 'k_n' in df_mnist.columns:
        for k in sorted(df_mnist['bottleneck'].unique()):
            sel = df_mnist[df_mnist['bottleneck'] == k]
            if sel.empty:
                continue
            plt.figure()
            for est in ests:
                tmp = sel[sel['estimator'] == est].groupby('k_n')['estimate'].agg(['mean', 'std']).reset_index()
                if tmp.empty:
                    continue
                plt.errorbar(tmp['k_n'], tmp['mean'], yerr=tmp['std'], marker='o', label=est, color=colors[est])
            plt.xlabel('k_n')
            plt.ylabel('estimate')
            plt.title(f'real k sensitivity (bottleneck={k})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'real_k_sensitivity_k_{k}.png'))

    # 12. Stability over k (real)
    try:
        plt.figure()
        for est in ests:
            tmp = df_mnist[df_mnist['estimator'] == est].groupby(['bottleneck', 'k_n'])['estimate'].mean().unstack()
            if tmp.empty:
                continue
            stability = tmp.max(axis=1) - tmp.min(axis=1)
            plt.plot(stability.index, stability.values, marker='o', label=est, color=colors[est])
        plt.xlabel('bottleneck')
        plt.ylabel('stability (max_k - min_k)')
        plt.title('real stability vs bottleneck')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'real_stability_vs_k.png'))
    except Exception:
        pass

    # 13. Sample size sensitivity: if sample size info present
    if 'n' in df_mnist.columns:
        for k in sorted(df_mnist['bottleneck'].unique()):
            sel = df_mnist[df_mnist['bottleneck'] == k]
            if sel.empty:
                continue
            plt.figure()
            for est in ests:
                tmp = sel[sel['estimator'] == est].groupby('n')['estimate'].agg(['mean', 'std']).reset_index()
                if tmp.empty:
                    continue
                plt.errorbar(tmp['n'], tmp['mean'], yerr=tmp['std'], marker='o', label=est, color=colors[est])
            plt.xlabel('n')
            plt.ylabel('estimate')
            plt.title(f'real sample size effect (bottleneck={k})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'real_sample_size_effect_k_{k}.png'))

    # 14. Histograms for synthetic are generated earlier per-estimator/d in separate scripts if needed.

    # 15. Scatter: estimate vs true for all points
    try:
        plt.figure()
        for est in ests:
            sub = df_syn[df_syn['estimator'] == est]
            if sub.empty:
                continue
            plt.scatter(sub['d'], sub['estimate'], label=est, alpha=0.6, color=colors[est])
        plt.xlabel('true d')
        plt.ylabel('estimate')
        plt.title('scatter: estimate vs true (all replicates)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'synthetic_scatter_all.png'))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'small', 'final'], default='smoke')
    parser.add_argument('--cpu', action='store_true', help='force CPU mode for training')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--base-seed', type=int, default=None, help='optional base seed override')
    args = parser.parse_args()

    # default configurations (mirror the provided config)
    if args.mode == 'smoke':
        syn_config = {
            'n_samples': 300,
            'intrinsic_dims': [2, 5],
            'manifolds': ['sphere', 'torus'],
            'noise_levels': [0.0, 0.05],
            'R': 2,
            'neighbor_grid_K': [5, 10],
            'methods': ['levina-bickel', 'twonn', 'corrint', 'danco', 'mind', 'fisher'],
            'base_seed': 0,
        }
        mnist_config = {
            'mnist_subset_size': 2000,
            'bottleneck_dims': [5, 10],
            'R': 1,
            'epochs': 5,
            'batch_size': 128,
            'neighbor_grid_K': [5, 10],
            'methods': ['levina-bickel', 'twonn', 'corrint', 'danco', 'mind', 'fisher'],
            'data_dir': 'data',
            'base_seed': 0,
        }
    elif args.mode == 'small':
        # Small experiment: development / debugging trends (user-provided config)
        syn_config = {
            'n_samples': 1000,
            'intrinsic_dims': [2, 5, 10, 15],
            'manifolds': ['sphere', 'torus'],
            'noise_levels': [0.0, 0.05, 0.1, 0.2],
            'R': 5,
            'neighbor_grid_K': [5, 10, 20, 30],
            'methods': ['levina-bickel', 'twonn', 'corrint', 'danco', 'mind', 'fisher'],
            'base_seed': 0,
        }
        mnist_config = {
            'mnist_subset_size': 60000,
            'bottleneck_dims': [5, 10, 20, 40],
            'R': 3,
            'epochs': 25,
            'batch_size': 128,
            'neighbor_grid_K': [5, 10, 20],
            'methods': ['levina-bickel', 'twonn', 'corrint', 'danco', 'mind', 'fisher'],
            'data_dir': 'data',
            'base_seed': 0,
        }
    # allow environment or CLI override for base seed (useful for Slurm arrays)
    env_base = os.environ.get('BASE_SEED')
    if env_base is not None:
        try:
            b = int(env_base)
            syn_config['base_seed'] = b
            mnist_config['base_seed'] = b
        except Exception:
            pass
    if args.base_seed is not None:
        syn_config['base_seed'] = args.base_seed
        mnist_config['base_seed'] = args.base_seed
    else:
        raise NotImplementedError('Only smoke mode implemented in this script')

    # choose output filenames; write figures into results/figs (overwrite existing figs)
    synthetic_out = f'results/synthetic_{args.mode}.csv'
    mnist_out = f'results/mnist_{args.mode}.csv'

    print(f'Running synthetic {args.mode} experiments...')
    df_syn = run_synthetic(syn_config, out_csv=synthetic_out, max_workers=args.workers)
    print(f'Saved {synthetic_out}')

    print(f'Running MNIST autoencoder {args.mode} experiments...')
    df_mnist = run_mnist_autoencoder(mnist_config, out_csv=mnist_out, run_train=True)
    print(f'Saved {mnist_out}')

    print('Making basic plots...')
    # overwrite figures in results/figs
    make_basic_plots(df_syn, out_dir='results/figs')
    # full suite
    try:
        make_full_synthetic_plots(df_syn, out_dir='results/figs')
    except Exception as e:
        print('Failed to make full synthetic plots:', e)
    try:
        make_full_mnist_plots(df_mnist, out_dir='results/figs')
    except Exception as e:
        print('Failed to make full MNIST plots:', e)
    print('Saved plots under results/figs')


if __name__ == '__main__':
    main()
