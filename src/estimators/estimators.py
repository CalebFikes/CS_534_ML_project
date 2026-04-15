"""Intrinsic-dimension estimators and wrappers.

This module provides faithful implementations for Levina-Bickel (MLE), TwoNN,
and a correlation-integral estimator. It will use `scikit-dimension` (skdim)
when available for more advanced estimators (DANCo, MiND). The API is a
simple `estimate(X, method, **kwargs)` function.
"""
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from sklearn.neighbors import NearestNeighbors
try:
    from .faiss_helpers import FAISS_AVAILABLE, faiss_knn_distances
except Exception:
    FAISS_AVAILABLE = False
    faiss_knn_distances = None
from sklearn.linear_model import LinearRegression

try:
    import skdim
    from skdim import id
    SKDIM_AVAILABLE = True
except Exception:
    SKDIM_AVAILABLE = False

try:
    from .masked_ae import masked_ae_estimate
except Exception:
    masked_ae_estimate = None

def _kneighbors_distances(X, k):
    # Prefer FAISS if available for speed
    if FAISS_AVAILABLE and faiss_knn_distances is not None:
        try:
            use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
            D, I = faiss_knn_distances(X.astype('float32'), k + 1, use_gpu=use_gpu)
            # D is squared L2 distances from FAISS; convert to sqrt
            D = np.sqrt(np.maximum(D, 0.0))
            return D[:, 1:]
        except Exception:
            pass
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    dists, _ = nn.kneighbors(X)
    # drop self-distance 0
    return dists[:, 1:]

def levina_bickel_mle(X, k=10):
    """Levina-Bickel MLE intrinsic dimension estimator.

    Implements the estimator from Levina & Bickel (2005). Returns a scalar d_hat.
    """
    n, D = X.shape
    if k >= n:
        raise ValueError("k must be < n")
    dists = _kneighbors_distances(X, k)
    # T_j are distances to j-th neighbor; T_k is last column
    T_k = dists[:, -1]
    logs = np.log(T_k[:, None] / dists[:, :-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_local = (np.mean(logs, axis=1))
        inv_local = np.where(inv_local <= 0, np.nan, inv_local)
        d_local = 1.0 / inv_local
    # average over points, ignoring nan
    return np.nanmean(d_local)

def twonn(X):
    """TwoNN estimator (Facco et al., 2017)."""
    dists = _kneighbors_distances(X, 2)
    T1 = dists[:, 0]
    T2 = dists[:, 1]
    eps = 1e-12
    mu = T2 / (T1 + eps)
    logs = np.log(mu)
    # mask non-finite values
    mask = np.isfinite(logs)
    if not np.any(mask):
        return float('nan')
    logs = logs[mask]
    mean_log = np.mean(logs)
    if not np.isfinite(mean_log) or mean_log == 0:
        return float('nan')
    return float(1.0 / mean_log)

def correlation_integral(X, n_r=20, r_min_quantile=0.01, r_max_quantile=0.2):
    """Estimate correlation dimension by scaling of C(r).

    Returns the estimated slope (dimension) using linear regression on the
    log-log relation for radii between specified quantiles of pairwise distances.
    """
    n = X.shape[0]
    from scipy.spatial.distance import pdist
    # determine radii from pairwise distances quantiles (use pdist for quantiles)
    if n < 2:
        return np.nan
    Dpairs = pdist(X)
    if len(Dpairs) == 0:
        return np.nan
    r_min = np.quantile(Dpairs, r_min_quantile)
    r_max = np.quantile(Dpairs, r_max_quantile)
    if r_min <= 0:
        r_min = np.nextafter(0, 1)
    rs = np.linspace(r_min, r_max, n_r)
    ns = []
    # If FAISS is available, use its range_search to compute counts efficiently
    if FAISS_AVAILABLE and 'faiss_knn_distances' in globals():
        try:
            from .faiss_helpers import faiss_range_counts
            use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
            counts = faiss_range_counts(X, rs, use_gpu=use_gpu)
            for c in counts:
                C = (2.0 * c) / (n * (n - 1))
                ns.append(C)
        except Exception:
            # fallback to pairwise counting
            for r in rs:
                C = np.sum(Dpairs < r) * 2.0 / (n * (n - 1))
                ns.append(C)
    else:
        for r in rs:
            C = np.sum(Dpairs < r) * 2.0 / (n * (n - 1))
            ns.append(C)
        # compute radii between a small value and max pairwise distance
        # get an approximate maximum distance using a few neighbors
        D_approx = _kneighbors_distances(X, min(10, n - 1))
        maxd = float(np.max(D_approx))
        if not np.isfinite(maxd) or maxd <= 0:
            return float('nan')
        radii = np.logspace(np.log10(1e-6), np.log10(maxd), n_r)
        counts = np.array([np.sum(Dpairs < r) for r in radii], dtype=float)
        # correlation integral C(r) ~ counts / (n*(n-1)/2)
        denom = (n * (n - 1) / 2)
        if denom <= 0:
            return float('nan')
        C = counts / (denom + 1e-12)
        # remove zero or non-finite entries before log
        valid = (C > 0) & np.isfinite(C) & np.isfinite(radii)
        if np.sum(valid) < 2:
            return float('nan')
        logs = np.log(radii[valid])
        logC = np.log(C[valid])
        slope, intercept = np.polyfit(logs, logC, 1)
        return float(slope)

def danco_wrapper(X):
    if not SKDIM_AVAILABLE:
        raise RuntimeError("scikit-dimension (skdim) is required for DANCo")
    estimator = id.DANCo()
    out = estimator.fit_transform(X)
    try:
        return float(np.asarray(out).item())
    except Exception:
        arr = np.asarray(out)
        return float(arr.mean())


def local_pca_wrapper(X):
    """Wrapper for scikit-dimension's local PCA (LPCA) estimator.

    Tries multiple common attribute names for the estimator class exposed by
    `skdim.id`. Returns a scalar estimate.
    """
    if not SKDIM_AVAILABLE:
        raise RuntimeError("scikit-dimension (skdim) is required for LPCA")

    # possible attribute names in different skdim versions
    candidates = ['LPCA', 'lPCA', 'LocalPCA', 'Local_PCA']
    EstClass = None
    for name in candidates:
        if hasattr(id, name):
            EstClass = getattr(id, name)
            break
    if EstClass is None:
        # fallback: try to find any class with 'LPCA' in its name
        for attr in dir(id):
            if 'LPCA' in attr.upper() or 'LOCAL' in attr.upper() and 'PCA' in attr.upper():
                EstClass = getattr(id, attr)
                break
    if EstClass is None:
        raise RuntimeError("LPCA estimator not found in skdim (checked common names)")

    estimator = EstClass()
    out = estimator.fit_transform(X)
    try:
        return float(np.asarray(out).item())
    except Exception:
        arr = np.asarray(out)
        return float(arr.mean())

def mind_wrapper(X):
    if not SKDIM_AVAILABLE:
        raise RuntimeError("scikit-dimension (skdim) is required for MiND")
    # skdim exposes MiND_ML (MiND maximum-likelihood) as MiND_ML
    if hasattr(id, 'MiND_ML'):
        estimator = id.MiND_ML()
    elif hasattr(id, 'MiND'):
        estimator = id.MiND()
    else:
        raise RuntimeError("MiND estimator not found in skdim")
    out = estimator.fit_transform(X)
    try:
        return float(np.asarray(out).item())
    except Exception:
        arr = np.asarray(out)
        return float(arr.mean())

def fisher_separability_placeholder(X):
    """Placeholder for Fisher separability estimator.

    Returns NaN but will not break downstream code. Replace with faithful
    implementation when available.
    """
    # try to use skdim's FisherS if available
    if SKDIM_AVAILABLE and hasattr(id, 'FisherS'):
        estimator = id.FisherS()
        out = estimator.fit_transform(X)
        try:
            return float(np.asarray(out).item())
        except Exception:
            return float(np.asarray(out).mean())
    return float('nan')

def estimate(X, method='levina-bickel', **kwargs):
    methods = {
        'levina-bickel': levina_bickel_mle,
        'twonn': twonn,
        'local-pca': local_pca_wrapper,
        'lPCA': local_pca_wrapper,
        'danco': danco_wrapper,
        'mind': mind_wrapper,
        'fisher': fisher_separability_placeholder,
        'masked-ae': masked_ae_estimate,
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    return methods[method](X, **kwargs) if kwargs else methods[method](X)
