from src.experiments.run_experiments import run_synthetic

if __name__ == '__main__':
    cfg = {
        'n_samples': 300,
        'intrinsic_dims': [2, 5],
        'manifolds': ['sphere', 'torus'],
        'noise_levels': [0.0, 0.05],
        'R': 2,
        'neighbor_grid_K': [5, 10],
        'methods': ['levina-bickel', 'twonn', 'corrint', 'danco', 'mind', 'fisher'],
        'base_seed': 0,
    }
    run_synthetic(cfg, out_csv='results/synthetic_smoke.csv', max_workers=2)
    # generate full plots for synthetic results
    from src.experiments.run_experiments import make_full_synthetic_plots
    import pandas as pd
    df = pd.read_csv('results/synthetic_smoke.csv')
    make_full_synthetic_plots(df, out_dir='results/figs')
    print('synthetic run complete')
