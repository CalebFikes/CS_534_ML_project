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
    # use standard deviation as error bars and mean as marker; stagger x per estimator
    import numpy as _np
    agg = df.groupby(['estimator', kcol])[value_col].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(8,6))
    estimators = sorted(agg['estimator'].unique())
    for i, est in enumerate(estimators):
        g = agg[agg['estimator']==est]
        if g.empty:
            continue
        y = g['mean'].values
        yerr = g['std'].values
        # small stagger so error bars don't overlap
        offset = (i - (len(estimators)-1)/2.0) * 0.08
        xs = g[kcol].values + offset
        plt.errorbar(xs, y, yerr=yerr, marker='o', markersize=4, capsize=3, linestyle='-', label=est)
    plt.xlabel(kcol)
    plt.ylabel(value_col)
    plt.title(f'{title_prefix}{value_col} vs {kcol}')
    plt.legend(fontsize='small', loc='outside right') #test
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
            for i, est in enumerate(estimators):
                if est in no_k_estimators:
                    est_df = cur[cur['estimator'] == est]
                else:
                    est_df = cur_k[cur_k['estimator'] == est]
                if est_df.empty:
                    continue
                # use mean and std as errorbars
                grp = est_df.groupby(true_col)[value_col].agg(['mean','std']).reset_index()
                if grp.empty:
                    continue
                plotted_any = True
                plotted_x.append(grp[true_col].values)
                y = grp['mean'].values
                yerr = grp['std'].values
                # stagger x slightly by estimator index so error bars don't fully overlap
                offset = (i - (len(estimators)-1)/2.0) * 0.06
                xs = grp[true_col].values + offset
                ax.errorbar(xs, y, yerr=yerr, marker='o', markersize=4, capsize=3, label=est, color=color_map.get(est))

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
            x = bbox.x0 #+ bbox.width/2.0
            # place header at the top border of the axes
            y = bbox.y1 + 0.05
            fig.text(x, y, f'sigma={sigma}', ha='center', va='top', fontsize=10)

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
        # place legend as a single-column right-hand column
        fig.subplots_adjust(right=0.78)
        fig.legend(list(leg_map.values()), list(leg_map.keys()), loc='center left', bbox_to_anchor=(0.82, 0.5), fontsize=10, ncol=1)

    plt.suptitle(f'{manifold}: {value_col} vs {true_col}')
    plt.tight_layout(rect=[0, 0, 0.86, 0.94])
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', outpath)


def plot_mnist_grid(df, outpath, true_col='bottleneck', value_col='estimate'):
    df = df.copy()
    df['estimator'] = df['estimator'].map(CANONICAL_EST_NAMES).fillna(df['estimator'])
    if df.empty:
        print('No MNIST data; skipping', outpath)
        return

    kcol = 'k_n' if 'k_n' in df.columns else ('k' if 'k' in df.columns else None)
    ks = sorted(df[kcol].unique()) if kcol is not None else [None]
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
            if kval is not None and kcol is not None:
                cur_k = cur[cur[kcol] == kval]
            else:
                cur_k = cur

            plotted_any = False
            plotted_x = []
            for i, est in enumerate(estimators):
                if est in no_k_estimators:
                    est_df = cur[cur['estimator'] == est]
                else:
                    est_df = cur_k[cur_k['estimator'] == est]
                if est_df.empty:
                    continue
                # use mean and std as errorbars
                grp = est_df.groupby(true_col)[value_col].agg(['mean','std']).reset_index()
                if grp.empty:
                    continue
                plotted_any = True
                plotted_x.append(grp[true_col].values)
                y = grp['mean'].values
                yerr = grp['std'].values
                offset = (i - (len(estimators)-1)/2.0) * 0.06
                xs = grp[true_col].values + offset
                ax.errorbar(xs, y, yerr=yerr, marker='o', markersize=4, capsize=3, label=est, color=color_map.get(est))

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
        x = bbox.x0 #+ bbox.width / 2.0
        # place header at the top border of the axes
        y = bbox.y1 + 0.05
        fig.text(x, y, f'sigma={sigma}', ha='center', va='top', fontsize=10)

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
        # place legend as a single-column right-hand column (match plot_est_vs_true_grid)
        fig.subplots_adjust(right=0.78)
        fig.legend(list(leg_map.values()), list(leg_map.keys()), loc='center left', bbox_to_anchor=(0.82, 0.5), fontsize=10, ncol=1)

    plt.suptitle(f'MNIST: {value_col} vs {true_col}')
    plt.tight_layout(rect=[0, 0, 0.86, 0.94])
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', outpath)


def plot_k_tracks(df, outpath, true_col='d', value_col='estimate'):
    """Plot estimate vs k (or k_n) for each estimator; non-k methods get horizontal lines.
    Adds a dotted horizontal line at the baseline true estimate computed from sigma==0 and max true_col."""
    if df is None or df.empty:
        print('No data for k-tracks; skipping', outpath)
        return
    df = df.copy()
    df['estimator'] = df['estimator'].map(CANONICAL_EST_NAMES).fillna(df['estimator'])
    kcol = 'k' if 'k' in df.columns else ('k_n' if 'k_n' in df.columns else None)
    if kcol is None:
        print('No k column to plot k-tracks; skipping', outpath)
        return

    estimators = sorted(df['estimator'].unique())
    cmap = plt.get_cmap('tab10')
    color_map = {est: cmap(i % 10) for i, est in enumerate(estimators)}
    no_k_estimators = set(['FisherS', 'SMAE', 'TwoNN'])
    plt.figure(figsize=(8,6))

    # determine baseline(s): if `true_col` (e.g. `bottleneck`) is present,
    # use its unique values as dotted baselines (useful for MNIST). Otherwise
    # fall back to the legacy heuristic (sigma==0 / max true) to compute a
    # single baseline value.
    baselines = []
    try:
        if true_col in df.columns:
            unique_trues = sorted(df[true_col].dropna().unique())
            # use the true values themselves as baselines
            baselines = [float(v) for v in unique_trues]
        else:
            if 'sigma' in df.columns:
                zeros = df[df['sigma'] == 0]
                if not zeros.empty:
                    max_true = zeros[true_col].max() if true_col in zeros.columns else None
                    if max_true is not None:
                        sel = zeros[zeros[true_col] == max_true]
                        if not sel.empty and value_col in sel.columns:
                            baselines = [float(sel[value_col].mean())]
                    if not baselines and value_col in zeros.columns:
                        baselines = [float(zeros[value_col].mean())]
    except Exception:
        baselines = []

    ks = sorted(df[kcol].dropna().unique())
    if not ks:
        print('No k values for k-tracks; skipping', outpath)
        return
    mid_x = float(sum(ks)) / len(ks)
    # diagnostic: check for extreme estimator values (helps spot DANCo inflation)
    try:
        max_true_overall = float(df[true_col].max()) if true_col in df.columns else None
    except Exception:
        max_true_overall = None
    for i, est in enumerate(estimators):
        cur = df[df['estimator'] == est]
        offset = (i - (len(estimators)-1)/2.0) * 0.12
        if est in no_k_estimators:
            if value_col in cur.columns:
                mean_val = cur[value_col].mean()
                std_val = cur[value_col].std()
                # horizontal comparison line: SOLID for non-k methods (baseline is the only dotted line)
                plt.hlines(mean_val, xmin=min(ks)-0.4, xmax=max(ks)+0.4, linestyles='-', linewidth=1.2, label=est, color=color_map.get(est))
                # constant error bar marker at mid position (staggered)
                plt.errorbar([mid_x + offset], [mean_val], yerr=[std_val], fmt='o', color=color_map.get(est), capsize=3, markersize=4)
                # warn on extreme values
                try:
                    if max_true_overall is not None and mean_val > max(3*max_true_overall, 1000):
                        print(f'WARNING: estimator {est} has unusually large mean {mean_val:.1f} (may indicate inflation)')
                except Exception:
                    pass
        else:
            grp = cur.groupby(kcol)[value_col].agg(['mean','std']).reset_index()
            if grp.empty:
                continue
            x = grp[kcol].values + offset
            y = grp['mean'].values
            yerr = grp['std'].values
            plt.errorbar(x, y, yerr=yerr, marker='o', markersize=4, capsize=3, linestyle='-', label=est, color=color_map.get(est))
            # check for inflated means
            try:
                mval = float(grp['mean'].max())
                if max_true_overall is not None and mval > max(3*max_true_overall, 1000):
                    print(f'WARNING: estimator {est} has unusually large value {mval:.1f} in k-tracks')
            except Exception:
                pass

    # draw baselines:
    # - For MNIST (true_col == 'bottleneck') draw dashed lines at 10 and 20 (hard-coded).
    # - For synthetic data draw a single baseline at the maximum true value (if available).
    if true_col == 'bottleneck':
        for b in (10, 20):
            plt.axhline(b, color='k', linestyle=':', linewidth=1.0, label='_nolegend_')
    else:
        if baselines:
            # use single baseline at max true value for synthetic k-tracks
            try:
                b = max(baselines)
                plt.axhline(b, color='k', linestyle=':', linewidth=1.0, label='_nolegend_')
            except Exception:
                for b in baselines:
                    plt.axhline(b, color='k', linestyle=':', linewidth=1.0, label='_nolegend_')

    plt.xlabel(kcol)
    plt.ylabel(value_col)
    plt.title(f'Estimate vs {kcol}')
    # place a single-column legend on the right with minimal spacing
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        fig = plt.gcf()
        fig.subplots_adjust(right=0.78)
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.82, 0.5), fontsize='small', ncol=1)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
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
            # per-manifold plots: grid and k-tracks only (no k-sensitivity plots)
            # grids
            if 'd' in syn.columns:
                plot_est_vs_true_grid(syn, 'sphere', figs / 'sphere_est_vs_d_grid.png', true_col='d', value_col=valcol)
                plot_est_vs_true_grid(syn, 'torus', figs / 'torus_est_vs_d_grid.png', true_col='d', value_col=valcol)
                # additional k-tracks plots: estimate vs k for each manifold
                try:
                    plot_k_tracks(syn[syn['manifold']=='sphere'], figs / 'sphere_k_tracks.png', true_col='d', value_col=valcol)
                except Exception:
                    pass
                try:
                    plot_k_tracks(syn[syn['manifold']=='torus'], figs / 'torus_k_tracks.png', true_col='d', value_col=valcol)
                except Exception:
                    pass
    else:
        print('No synthetic combined CSV at', syn_comb)

    mnist_comb = RES / f'mnist_{size}_combined.csv'
    if mnist_comb.exists():
        mn = pd.read_csv(mnist_comb)
        valcol = 'd_hat' if 'd_hat' in mn.columns else ('estimate' if 'estimate' in mn.columns else None)
        if valcol is None:
            print('No value column in mnist combined; skipping MNIST plots')
        else:
            # MNIST: only grid and k-tracks (no k-sensitivity plot)
            if 'bottleneck' in mn.columns:
                plot_mnist_grid(mn, figs / 'mnist_est_vs_bottleneck_grid.png', true_col='bottleneck', value_col=valcol)
                # MNIST k-tracks (use k_n if present)
                try:
                    plot_k_tracks(mn, figs / 'mnist_k_tracks.png', true_col='bottleneck', value_col=valcol)
                except Exception:
                    pass
    else:
        print('No mnist combined CSV at', mnist_comb)

    # Cleanup: ensure only the canonical six plots remain in the figs directory.
    try:
        keep = set([
            'sphere_est_vs_d_grid.png', 'sphere_k_tracks.png',
            'torus_est_vs_d_grid.png', 'torus_k_tracks.png',
            'mnist_est_vs_bottleneck_grid.png', 'mnist_k_tracks.png',
        ])
        for fn in os.listdir(str(figs)):
            if not fn.lower().endswith('.png'):
                continue
            if fn in keep:
                continue
            fpath = figs / fn
            try:
                os.remove(fpath)
                print('Removed extra figure', fpath)
            except Exception:
                pass
    except Exception:
        pass


if __name__ == '__main__':
    main()
