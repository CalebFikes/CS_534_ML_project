Final experiment spec (exact values)

- Mode: final
- Manifolds: [sphere, torus]
- n_samples (synthetic): 300
- Neighbor grid K (final): [3, 4, 5, 6, 7, 8, 10, 12, 15, 18]
- Noise levels (final): 10 values evenly spaced on [0, 1] inclusive -> [0.0, 0.1111111111, 0.2222222222, 0.3333333333, 0.4444444444, 0.5555555556, 0.6666666667, 0.7777777778, 0.8888888889, 1.0]
- Intrinsic dims / bottlenecks (final): [5, 6, 7, 9, 11, 13, 15]
- Replicates R (final): 10
- MNIST AE training epochs (final): 25
- MNIST AE batch size: 128 (unchanged)
- Masked-AE estimator training: lr same as default (1e-3), hidden dim same as AE default, batch_size same as AE default, epochs=10
- Masked-AE lambda grid: 10 log-spaced values in (0.1, 10) (inclusive endpoints used)
- AE architecture: use existing AE architecture in `src/models/train_autoencoder.py` (no change)
- Estimators included: ['levina-bickel', 'twonn', 'lPCA', 'danco', 'mind', 'fisher', 'masked-ae']
- CSV schema: rows contain estimator, base_seed, manifold, d, sigma, k, n, estimate, seed, error (synthetic) and estimator, base_seed, bottleneck, k, sigma, estimate, seed, error (mnist)

Notes:
- Use `cs_534_gpu` conda environment Python for runs if it contains `faiss` and `skdim`; otherwise fallback plan is to install compatible `torch` in `cs_534_final_gpu` per user instruction.
- This file records the exact final-size parameterization only (as requested).
