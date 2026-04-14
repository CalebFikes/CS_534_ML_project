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
    # simple d-dimensional torus: sample angles and map to 2d space per circle
    # For convenience return shape (n, 2*d)
    angles = rng.uniform(0, 2 * np.pi, size=(n, d))
    R = 2.0
    r = 0.5
    coords = []
    for j in range(d):
        x = (R + r * np.cos(angles[:, j])) * np.cos(angles[:, j])
        y = (R + r * np.cos(angles[:, j])) * np.sin(angles[:, j])
        coords.append(x)
        coords.append(y)
    return np.vstack(coords).T


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
