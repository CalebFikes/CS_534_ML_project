import numpy as np
from src.data import generators

def test_sample_sphere():
    X = generators.sample_sphere(2, 100, random_state=0)
    assert X.shape == (100, 3)
    # norms should be ~1
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms.mean(), 1.0, atol=1e-6)

def test_sample_torus():
    X = generators.sample_torus(3, 50, random_state=0)
    assert X.shape == (50, 6)

def test_embedding_and_noise():
    X = generators.sample_sphere(2, 20, random_state=1)
    Y = generators.embed_via_random_orthonormal(X, D=10, random_state=2)
    assert Y.shape == (20, 10)
    Z = generators.add_orthogonal_noise(Y, sigma=0.1, random_state=3)
    assert Z.shape == Y.shape
