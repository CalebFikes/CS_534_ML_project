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
import inspect
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


def _kneighbors(X, k):
    """Return (distances, indices) for k nearest neighbors (excluding self).

    Distances are L2 distances (not squared). Indices are integers.
    """
    # Prefer FAISS if available
    if FAISS_AVAILABLE and faiss_knn_distances is not None:
        try:
            use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
            D, I = faiss_knn_distances(X.astype('float32'), k + 1, use_gpu=use_gpu)
            # D is squared L2 distances from FAISS; convert to sqrt
            D = np.sqrt(np.maximum(D, 0.0))
            return D[:, 1:], I[:, 1:]
        except Exception:
            pass
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    dists, inds = nn.kneighbors(X)
    return dists[:, 1:], inds[:, 1:]


def _call_fit_transform_pw_with_neighbors(est, X, k, neigh_idx):
    """Attempt to call a pointwise fit_transform with precomputed neighbor indices.

    Tries a few common keyword names for neighbor arguments. If none match
    the estimator's signature, falls back to calling the estimator's
    pointwise API with `n_neighbors` or the global `fit_transform`.
    Returns the estimator output.
    """
    # ensure integer indices
    neigh_idx = np.asarray(neigh_idx, dtype=np.int64)
    # try common kwarg names
    candidate_kw = ['neighbors', 'nbrs', 'neighbor_indices', 'indices', 'neigh_idx', 'nbr_idx', 'neighbors_idx']
    func = None
    if hasattr(est, 'fit_transform_pw'):
        func = est.fit_transform_pw
    elif hasattr(est, 'fit_transform_pointwise'):
        func = est.fit_transform_pointwise

    if func is None:
        # no pointwise API: fallback to global fit_transform
        return est.fit_transform(X)

    sig = None
    try:
        sig = inspect.signature(func)
        params = sig.parameters
    except Exception:
        params = {}

    # try to find a kw that matches
    for kw in candidate_kw:
        if kw in params:
            try:
                return func(X, **{kw: neigh_idx})
            except Exception:
                pass

    # try combined call with n_neighbors + neighbors
    try:
        return func(X, n_neighbors=int(k), neighbors=neigh_idx)
    except Exception:
        pass

    # try calling with only n_neighbors
    try:
        return func(X, n_neighbors=int(k))
    except Exception:
        # last resort: global fit_transform
        return est.fit_transform(X)

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

def danco_wrapper(X, k=None, D = 50):
    """DANCo wrapper. Accepts optional neighborhood size `k`.

    If `k` is provided (passed via kwargs from `estimate`), it is forwarded
    to the DANCo constructor.
    """
    def _inner(X, k=10):
        if not SKDIM_AVAILABLE:
            raise RuntimeError("scikit-dimension (skdim) is required for DANCo")
        # Sanitize input: ensure numeric ndarray and replace non-finite values
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                print('DANCo: warning - non-finite values detected and replaced in input')
            except Exception:
                pass
        estimator = id.DANCo(k=int(k), D=int(D))
        # prefer pointwise API with precomputed neighbors when available
        try:
            dists, inds = _kneighbors(X, int(k))
            out = _call_fit_transform_pw_with_neighbors(estimator, X, int(k), inds)
        except Exception:
            out = estimator.fit_transform(X)
        try:
            return float(np.asarray(out).item())
        except Exception:
            arr = np.asarray(out)
            return float(arr.mean())

    # forward optional `k` to the inner implementation
    if k is None:
        return _inner(X)
    else:
        return _inner(X, k=int(k))


def local_pca_wrapper(X, k=None):
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

    def _inner(X, k=None):
        if not SKDIM_AVAILABLE:
            raise RuntimeError("scikit-dimension (skdim) is required for LPCA")

        est = EstClass()
        # If a neighborhood size `k` is supplied, use the pointwise API
        # and aggregate the pointwise estimates into a scalar (mean).
        if k is not None:
            try:
                dists, inds = _kneighbors(X, int(k))
                out = _call_fit_transform_pw_with_neighbors(est, X, int(k), inds)
                arr = np.asarray(out)
                return float(arr.mean())
            except Exception:
                # fallback to global estimator
                out = est.fit_transform(X)
                try:
                    return float(np.asarray(out).item())
                except Exception:
                    return float(np.asarray(out).mean())
        else:
            out = est.fit_transform(X)
            try:
                return float(np.asarray(out).item())
            except Exception:
                arr = np.asarray(out)
                return float(arr.mean())

    # forward optional neighborhood size if provided
    return _inner(X, k=int(k)) if k is not None else _inner(X)

def mind_wrapper(X, k=None):
    """MiND wrapper. Accepts optional `k` to control neighborhood size.

    If `k` is provided it is forwarded to the MiND constructor.
    """
    def _inner(X, k=20):
        if not SKDIM_AVAILABLE:
            raise RuntimeError("scikit-dimension (skdim) is required for MiND")
        # prefer MiND_ML if present
        if hasattr(id, 'MiND_ML'):
            Est = id.MiND_ML
        elif hasattr(id, 'MiND'):
            Est = id.MiND
        else:
            raise RuntimeError("MiND estimator not found in skdim")
        estimator = Est(k=int(k))
        try:
            dists, inds = _kneighbors(X, int(k))
            out = _call_fit_transform_pw_with_neighbors(estimator, X, int(k), inds)
        except Exception:
            out = estimator.fit_transform(X)
        try:
            return float(np.asarray(out).item())
        except Exception:
            arr = np.asarray(out)
            return float(arr.mean())

    # forward optional `k` to MiND
    if k is None:
        return _inner(X)
    else:
        return _inner(X, k=int(k))

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
        # accept canonical/display names as well
        'Levina-Bickel': levina_bickel_mle,
        'TwoNN': twonn,
        'DANCo': danco_wrapper,
        'MiND': mind_wrapper,
        'FisherS': fisher_separability_placeholder,
        'SMAE': masked_ae_estimate,
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    return methods[method](X, **kwargs) if kwargs else methods[method](X)
