#!/usr/bin/env python3
from src.experiments.run_experiments import make_full_synthetic_plots, make_full_mnist_plots, make_basic_plots
import pandas as pd

def main():
    syn = pd.read_csv('results/synthetic_small.csv')
    mn = pd.read_csv('results/mnist_small.csv')
    make_basic_plots(syn, 'results/figs')
    make_full_synthetic_plots(syn, 'results/figs')
    make_full_mnist_plots(mn, 'results/figs')
    print('Plots generated in results/figs')

if __name__ == '__main__':
    main()
