"""Lightweight data generators for smoke/small experiments.

These implementations are intentionally simple and deterministic-enough
for test runs. They are not optimized for research fidelity.
"""
import numpy as np


def sample_sphere(d, n, random_state=None):
    rng = np.random.default_rng(random_state)
    # sample n points in R^{d+1} and project to unit sphere
    X = rng.normal(size=(n, d + 1))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def sample_torus(d, n, random_state=None):
    rng = np.random.default_rng(random_state)
    theta = rng.uniform(0, 2*np.pi, size=(n, d))

    X = np.empty((n, 2*d))
    for j in range(d):
        X[:, 2*j]   = np.cos(theta[:, j])
        X[:, 2*j+1] = np.sin(theta[:, j])
    return X


def embed_via_random_orthonormal(X, D, random_state=None):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if D <= d:
        return X[:, :D]
    # create random orthonormal basis of size D x d and embed
    A = rng.normal(size=(D, d))
    # orthonormalize via QR
    Q, _ = np.linalg.qr(A)
    return X @ Q.T


def add_orthogonal_noise(X, sigma, random_state=None):
    if sigma is None or sigma == 0:
        return X
    rng = np.random.default_rng(random_state)
    noise = rng.normal(scale=sigma, size=X.shape)
    return X + noise
