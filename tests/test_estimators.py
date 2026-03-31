import numpy as np
from src.data.generators import sample_sphere
from src.estimators.estimators import levina_bickel_mle, twonn, correlation_integral, estimate

def test_levina_bickel():
    X = sample_sphere(2, 200, random_state=0)
    d = levina_bickel_mle(X, k=10)
    assert d > 0 and d < 10

def test_twonn():
    X = sample_sphere(3, 300, random_state=1)
    d = twonn(X)
    assert d > 0 and d < 20

def test_corrint():
    X = sample_sphere(2, 150, random_state=2)
    d = correlation_integral(X, n_r=10)
    assert d > 0 and d < 10

def test_estimate_dispatch():
    X = sample_sphere(1, 100, random_state=3)
    for m in ['levina-bickel', 'twonn', 'corrint', 'fisher']:
        val = estimate(X, method=m)
        assert np.isfinite(val) or np.isnan(val)
