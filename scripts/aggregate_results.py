"""Aggregate per-task CSVs produced by Slurm arrays and produce summaries.

Usage examples:
  python scripts/aggregate_results.py results/synthetic_smoke.task*.csv \
      --out-raw results/synthetic_smoke_combined.csv \
      --out-summary results/synthetic_smoke_summary.csv

This script concatenates matching CSVs, writes a combined raw CSV, and also
writes a summary CSV grouped by sensible columns (mean, std, count).
"""
import argparse
import glob
import os
import pandas as pd


def infer_group_cols(df):
    # prefer common groupings used by the plotting code
    if 'manifold' in df.columns:
        # synthetic: group by estimator, manifold, d, sigma, k
        cols = ['estimator', 'manifold', 'd', 'sigma']
        if 'k' in df.columns:
            cols = cols + ['k']
        return cols
    if 'bottleneck' in df.columns:
        # mnist: group by estimator and bottleneck
        cols = ['estimator', 'bottleneck']
        if 'k_n' in df.columns:
            cols = cols + ['k_n']
        return cols
    # fallback: group by estimator only
    return ['estimator']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('patterns', nargs='+', help='glob patterns for per-task CSVs')
    p.add_argument('--out-raw', default='results/combined_raw.csv')
    p.add_argument('--out-summary', default='results/combined_summary.csv')
    args = p.parse_args()

    paths = []
    for pat in args.patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print('No files matched; patterns:', args.patterns)
        return

    dfs = []
    for pth in paths:
        try:
            dfs.append(pd.read_csv(pth))
        except Exception as e:
            print('failed to read', pth, e)
    if not dfs:
        print('No valid CSVs to aggregate')
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(args.out_raw) or '.', exist_ok=True)
    combined.to_csv(args.out_raw, index=False)
    print('Wrote combined raw CSV to', args.out_raw)

    group_cols = infer_group_cols(combined)
    # compute mean, std, and count of estimates
    if 'estimate' in combined.columns:
        summary = combined.groupby(group_cols)['estimate'].agg(['mean', 'std', 'count']).reset_index()
        # if error column present, compute number of errors per group
        if 'error' in combined.columns:
            errcounts = combined.assign(err_flag=combined['error'].notnull() & (combined['error'] != '')).groupby(group_cols)['err_flag'].sum().reset_index()
            summary = summary.merge(errcounts, on=group_cols, how='left')
            summary = summary.rename(columns={'err_flag': 'error_count'})
        summary.to_csv(args.out_summary, index=False)
        print('Wrote aggregated summary CSV to', args.out_summary)
    else:
        print('No "estimate" column found; wrote only raw combined CSV')


if __name__ == '__main__':
    main()
