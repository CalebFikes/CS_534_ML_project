#!/usr/bin/env python3
"""Plot aggregated results for a given size (smoke/small/large/final).
Produces six plots into `results/figs_{size}`:
  - synthetic_k_sensitivity.png (all manifolds)
  - synthetic_k_sensitivity_sphere.png
  - synthetic_k_sensitivity_torus.png
  - sphere_est_vs_d_grid.png
  - torus_est_vs_d_grid.png
  - mnist_est_vs_bottleneck_grid.png

Usage: python scripts/plot_results.py --size large
"""
import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


# map internal estimator keys to canonical display names for plotting
CANONICAL_EST_NAMES = {
    'levina-bickel': 'Levina-Bickel',
    'twonn': 'TwoNN',
    'danco': 'DANCo',
    'mind': 'MiND',
    'fisher': 'FisherS',
    'masked-ae': 'SMAE',
}


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_k_sensitivity(df, outpath, value_col='estimate', title_prefix=''):
    df = df.copy()
    df['estimator'] = df['estimator'].map(CANONICAL_EST_NAMES).fillna(df['estimator'])
    if 'k' not in df.columns and 'k_n' not in df.columns:
        print('No k column in dataframe; skipping k-sensitivity:', outpath)
        return
    kcol = 'k' if 'k' in df.columns else 'k_n'
    grp = df.groupby(['estimator', kcol])[value_col].agg(['mean','sem']).reset_index()
    plt.figure(figsize=(8,6))
    for est in grp['estimator'].unique():
        g = grp[grp['estimator']==est]
        plt.errorbar(g[kcol], g['mean'], yerr=g['sem'], marker='o', label=est)
    plt.xlabel(kcol)
    plt.ylabel(value_col)
    plt.title(f'{title_prefix}{value_col} vs {kcol}')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print('Wrote', outpath)


def plot_est_vs_true_grid(df, manifold, outpath, true_col='d', value_col='estimate'):
    df = df.copy()
    df['estimator'] = df['estimator'].map(CANONICAL_EST_NAMES).fillna(df['estimator'])
    sub = df[df['manifold'] == manifold]
    if sub.empty:
        print('No rows for manifold', manifold)
        return

    ks = sorted(sub['k'].unique()) if 'k' in sub.columns else [None]
    sigmas = sorted(sub['sigma'].unique()) if 'sigma' in sub.columns else [None]

    nrow = len(ks)
    ncol = len(sigmas)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), squeeze=False)

    estimators = sorted(sub['estimator'].unique())
    cmap = plt.get_cmap('tab10')
    color_map = {est: cmap(i % 10) for i, est in enumerate(estimators)}
    no_k_estimators = set(['FisherS', 'SMAE', 'TwoNN'])

    for ri, kval in enumerate(ks):
        for ci, sigma in enumerate(sigmas):
            ax = axes[ri][ci]
            cur = sub.copy()
            if sigma is not None:
                cur = cur[cur['sigma'] == sigma]
            if kval is not None and 'k' in cur.columns:
                cur_k = cur[cur['k'] == kval]
            else:
                cur_k = cur

            plotted_any = False
            plotted_x = []
            for est in estimators:
                if est in no_k_estimators:
                    est_df = cur[cur['estimator'] == est]
                else:
                    est_df = cur_k[cur_k['estimator'] == est]
                if est_df.empty:
                    continue
                grp = est_df.groupby(true_col)[value_col].agg(['mean','sem']).reset_index()
                if grp.empty:
                    continue
                plotted_any = True
                plotted_x.append(grp[true_col].values)
                ax.errorbar(grp[true_col], grp['mean'], yerr=grp['sem'].fillna(0.0), marker='o', label=est, color=color_map.get(est))

            if not plotted_any:
                ax.axis('off')
                continue

            # identity line across plotted range
            if plotted_x:
                import numpy as _np
                xs = _np.concatenate(plotted_x)
                mn_x = float(xs.min())
                mx_x = float(xs.max())
                ax.plot([mn_x, mx_x], [mn_x, mx_x], '--', color='k')

            # row label on leftmost column
            if ci == 0 and kval is not None:
                ax.set_ylabel(f'k={kval}')

    # add column headers at figure level
    for ci, sigma in enumerate(sigmas):
        if sigma is None:
            continue
        ax0 = axes[0][ci]
        bbox = ax0.get_position()
        x = bbox.x0 + bbox.width/2.0
        y = bbox.y1 + 0.01
        fig.text(x, y, f'sigma={sigma}', ha='center', va='bottom', fontsize=10)

    # collect legend handles from all subplots (preserve order, unique labels)
    from collections import OrderedDict
    leg_map = OrderedDict()
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in leg_map:
                    leg_map[ll] = hh
    if leg_map:
        fig.subplots_adjust(right=0.82)
        fig.legend(list(leg_map.values()), list(leg_map.keys()), loc='center left', bbox_to_anchor=(0.86, 0.5), fontsize=10)

    plt.suptitle(f'{manifold}: {value_col} vs {true_col}')
    plt.tight_layout(rect=[0, 0, 0.82, 0.96])
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', outpath)


def plot_mnist_grid(df, outpath, true_col='bottleneck', value_col='estimate'):
    df = df.copy()
    df['estimator'] = df['estimator'].map(CANONICAL_EST_NAMES).fillna(df['estimator'])
    if df.empty:
        print('No MNIST data; skipping', outpath)
        return

    ks = sorted(df['k_n'].unique()) if 'k_n' in df.columns else [None]
    sigmas = sorted(df['sigma'].unique()) if 'sigma' in df.columns else [None]

    nrow = len(ks)
    ncol = len(sigmas)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), squeeze=False)

    estimators = sorted(df['estimator'].unique())
    cmap = plt.get_cmap('tab10')
    color_map = {est: cmap(i % 10) for i, est in enumerate(estimators)}
    no_k_estimators = set(['FisherS', 'SMAE', 'TwoNN'])

    for ri, kval in enumerate(ks):
        for ci, sigma in enumerate(sigmas):
            ax = axes[ri][ci]
            cur = df.copy()
            if sigma is not None:
                cur = cur[cur['sigma'] == sigma]
            if kval is not None and 'k_n' in cur.columns:
                cur_k = cur[cur['k_n'] == kval]
            else:
                cur_k = cur

            plotted_any = False
            plotted_x = []
            for est in estimators:
                if est in no_k_estimators:
                    est_df = cur[cur['estimator'] == est]
                else:
                    est_df = cur_k[cur_k['estimator'] == est]
                if est_df.empty:
                    continue
                grp = est_df.groupby(true_col)[value_col].agg(['mean','sem']).reset_index()
                if grp.empty:
                    continue
                plotted_any = True
                plotted_x.append(grp[true_col].values)
                ax.errorbar(grp[true_col], grp['mean'], yerr=grp['sem'].fillna(0.0), marker='o', label=est, color=color_map.get(est))

            if not plotted_any:
                ax.axis('off')
                continue

            if plotted_x:
                import numpy as _np
                xs = _np.concatenate(plotted_x)
                mn_x = float(xs.min())
                mx_x = float(xs.max())
                ax.plot([mn_x, mx_x], [mn_x, mx_x], '--', color='k')

            if ci == 0 and kval is not None:
                ax.set_ylabel(f'k={kval}')

    # add column headers
    for ci, sigma in enumerate(sigmas):
        if sigma is None:
            continue
        ax0 = axes[0][ci]
        bbox = ax0.get_position()
        x = bbox.x0 + bbox.width / 2.0
        y = bbox.y1 + 0.01
        fig.text(x, y, f'sigma={sigma}', ha='center', va='bottom', fontsize=10)

    # collect legend handles from all subplots (preserve order, unique labels)
    from collections import OrderedDict
    leg_map = OrderedDict()
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in leg_map:
                    leg_map[ll] = hh
    if leg_map:
        fig.subplots_adjust(right=0.78)
        fig.legend(list(leg_map.values()), list(leg_map.keys()), loc='center left', bbox_to_anchor=(0.82, 0.5), fontsize=10)

    plt.suptitle(f'MNIST: {value_col} vs {true_col}')
    plt.tight_layout(rect=[0, 0, 0.76, 0.96])
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', outpath)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='large')
    p.add_argument('--finalize', action='store_true', help='Aggregate per-task CSVs before plotting')
    args = p.parse_args()
    size = args.size
    figs = RES / f'figs_{size}'
    ensure_dir(figs)

    # If finalize requested, aggregate per-task CSVs into combined CSVs in results/
    if args.finalize:
        import glob, os

        def infer_group_cols(df):
            if 'manifold' in df.columns:
                cols = ['estimator', 'manifold', 'd', 'sigma']
                if 'k' in df.columns:
                    cols = cols + ['k']
                return cols
            if 'bottleneck' in df.columns:
                cols = ['estimator', 'bottleneck']
                if 'k_n' in df.columns:
                    cols = cols + ['k_n']
                return cols
            return ['estimator']

        def aggregate(patterns, out_raw, out_summary):
            paths = []
            for pat in patterns:
                paths.extend(sorted(glob.glob(pat)))
            if not paths:
                return False
            dfs = []
            import pandas as _pd
            for pth in paths:
                try:
                    dfs.append(_pd.read_csv(pth))
                except Exception:
                    pass
            if not dfs:
                return False
            combined = _pd.concat(dfs, ignore_index=True, sort=False)
            os.makedirs(os.path.dirname(out_raw) or '.', exist_ok=True)
            combined.to_csv(out_raw, index=False)
            group_cols = infer_group_cols(combined)
            if 'estimate' in combined.columns:
                summary = combined.groupby(group_cols)['estimate'].agg(['mean', 'std', 'count']).reset_index()
                if 'error' in combined.columns:
                    errcounts = combined.assign(err_flag=combined['error'].notnull() & (combined['error'] != '')).groupby(group_cols)['err_flag'].sum().reset_index()
                    summary = summary.merge(errcounts, on=group_cols, how='left')
                    summary = summary.rename(columns={'err_flag': 'error_count'})
                summary.to_csv(out_summary, index=False)
            return True

        # aggregate synthetic and mnist per-task CSVs for this size
        aggregate([str(RES / f'synthetic_{size}.task*.csv')], str(RES / f'synthetic_{size}_combined.csv'), str(RES / f'synthetic_{size}_summary.csv'))
        aggregate([str(RES / f'mnist_{size}.task*.csv')], str(RES / f'mnist_{size}_combined.csv'), str(RES / f'mnist_{size}_summary.csv'))

        # Fallback: some runs (smoke) write a single per-worker CSV without the .task suffix.
        # In that case copy the single raw CSV to the expected _combined.csv so plotting can proceed.
        import shutil
        syn_raw = RES / f'synthetic_{size}.csv'
        syn_comb = RES / f'synthetic_{size}_combined.csv'
        if not syn_comb.exists() and syn_raw.exists():
            shutil.copy(syn_raw, syn_comb)
            print('Copied', syn_raw, '->', syn_comb)

        mn_raw = RES / f'mnist_{size}.csv'
        mn_comb = RES / f'mnist_{size}_combined.csv'
        if not mn_comb.exists() and mn_raw.exists():
            shutil.copy(mn_raw, mn_comb)
            print('Copied', mn_raw, '->', mn_comb)

    syn_comb = RES / f'synthetic_{size}_combined.csv'
    if syn_comb.exists():
        syn = pd.read_csv(syn_comb)
        valcol = 'd_hat' if 'd_hat' in syn.columns else ('estimate' if 'estimate' in syn.columns else None)
        if valcol is None:
            print('No value column in synthetic combined; skipping plots')
        else:
            # per-manifold k sensitivity (one plot per manifold)
            plot_k_sensitivity(syn[syn['manifold']=='sphere'], figs / 'synthetic_k_sensitivity_sphere.png', value_col=valcol, title_prefix='Sphere: ')
            plot_k_sensitivity(syn[syn['manifold']=='torus'], figs / 'synthetic_k_sensitivity_torus.png', value_col=valcol, title_prefix='Torus: ')
            # grids
            if 'd' in syn.columns:
                plot_est_vs_true_grid(syn, 'sphere', figs / 'sphere_est_vs_d_grid.png', true_col='d', value_col=valcol)
                plot_est_vs_true_grid(syn, 'torus', figs / 'torus_est_vs_d_grid.png', true_col='d', value_col=valcol)
    else:
        print('No synthetic combined CSV at', syn_comb)

    mnist_comb = RES / f'mnist_{size}_combined.csv'
    if mnist_comb.exists():
        mn = pd.read_csv(mnist_comb)
        valcol = 'd_hat' if 'd_hat' in mn.columns else ('estimate' if 'estimate' in mn.columns else None)
        if valcol is None:
            print('No value column in mnist combined; skipping MNIST plots')
        else:
            plot_k_sensitivity(mn, figs / 'mnist_k_sensitivity.png', value_col=valcol, title_prefix='MNIST: ')
            if 'bottleneck' in mn.columns:
                plot_mnist_grid(mn, figs / 'mnist_est_vs_bottleneck_grid.png', true_col='bottleneck', value_col=valcol)
    else:
        print('No mnist combined CSV at', mnist_comb)


if __name__ == '__main__':
    main()
