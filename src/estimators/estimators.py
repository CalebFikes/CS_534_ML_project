import numpy as np
from sklearn.neighbors import NearestNeighbors
try:
    from .faiss_helpers import FAISS_AVAILABLE, faiss_knn_distances
except Exception:
    FAISS_AVAILABLE = False
    faiss_knn_distances = None

try:
    import skdim
    from skdim import id
    SKDIM_AVAILABLE = True
except Exception:
    print("NO SKDIM")
    SKDIM_AVAILABLE = False

def _kneighbors_distances(X, k):
    # Prefer FAISS if available for speed
    if FAISS_AVAILABLE and faiss_knn_distances is not None:
        try:
            D, I = faiss_knn_distances(X.astype('float32'), k + 1)
            # D is squared L2 distances from FAISS; convert to sqrt
            D = np.sqrt(np.maximum(D, 0.0))
            return D[:, 1:]
        except Exception:
            pass
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    dists, _ = nn.kneighbors(X)
    # drop self-distance 0
    return dists[:, 1:]

#Levina-Bickel LME estimator--implementation based off of LLM generation, but checked carefully
def levina_bickel_mle(X, k=10):
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

#use skdim implementations when avaliable:
def twonn_wrapper(X):
    if not SKDIM_AVAILABLE:
        raise RuntimeError("SkDim not available")
    
    estimator = id.TwoNN()
    out = estimator.fit_transform(X)
    return float(out)

def danco_wrapper(X):
    if not SKDIM_AVAILABLE:
        raise RuntimeError("SkDim not available")
    estimator = id.DANCo()
    out = estimator.fit_transform(X)
    
    return float(out)

def mind_wrapper(X):
    #scikit MiND_ML Implementation:
    if not SKDIM_AVAILABLE:
        raise RuntimeError("SkDim not available")
    estimator = id.MiND_ML()
   
    out = estimator.fit_transform(X)
    return float(out)

def fisher_wrapper(X):
    if not SKDIM_AVAILABLE:
        raise RuntimeError("SkDim not available")
    estimator = id.FisherS()
    
    out = estimator.fit_transform(X)
    return float(out)

def estimate(X, method='levina-bickel', **kwargs):
    methods = {
        'levina-bickel': levina_bickel_mle,
        'twonn': twonn_wrapper,
        'danco': danco_wrapper,
        'mind': mind_wrapper,
        'fisher': fisher_wrapper
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    
    if kwargs:
        return methods[method](X, **kwargs)
    else:
        return methods[method](X)
