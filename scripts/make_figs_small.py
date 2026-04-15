"""Generate plots for the 'small' experiment mode into results/figs_small

This helper uses existing CSVs if present. It avoids retraining AEs.
"""
import os
import pandas as pd
from src.experiments.run_experiments import make_basic_plots, make_full_synthetic_plots, make_full_mnist_plots

os.makedirs('results', exist_ok=True)
# prefer combined small synthetic CSV
syn_csv = 'results/synthetic_small_combined.csv'
if not os.path.exists(syn_csv):
    syn_csv = 'results/synthetic_small.csv'

mnist_csv = 'results/mnist_small.csv'
if not os.path.exists(mnist_csv):
    # fall back to smoke mnist if small not available
    mnist_csv = 'results/mnist_smoke_combined.csv' if os.path.exists('results/mnist_smoke_combined.csv') else 'results/mnist_smoke.csv'

print('Using synthetic CSV:', syn_csv)
print('Using mnist CSV:', mnist_csv)

df_syn = pd.read_csv(syn_csv) if os.path.exists(syn_csv) else None
if df_syn is None or df_syn.empty:
    print('No synthetic data available; aborting.')
else:
    out_dir = 'results/figs_small'
    make_basic_plots(df_syn, out_dir=out_dir)
    try:
        make_full_synthetic_plots(df_syn, out_dir=out_dir)
    except Exception as e:
        print('Failed full synthetic plots:', e)

if os.path.exists(mnist_csv):
    df_mnist = pd.read_csv(mnist_csv)
else:
    df_mnist = None

if df_mnist is None or df_mnist.empty:
    print('No MNIST results available to plot; skipping MNIST plots.')
else:
    out_dir = 'results/figs_small'
    try:
        make_full_mnist_plots(df_mnist, out_dir=out_dir)
    except Exception as e:
        print('Failed MNIST plots:', e)

print('Done; figures (if any) are in results/figs_small')
