try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

import numpy as np

def faiss_knn_distances(X, k, use_gpu=False):
    """Compute k nearest neighbor distances for rows of X using FAISS.

    Returns an (n x k) array of distances (sorted ascending) excluding self (so caller
    should request k+1 and drop first column if desired).
    """
    X = np.asarray(X).astype('float32')
    n, d = X.shape
    if not FAISS_AVAILABLE:
        raise RuntimeError('faiss is not installed')
    index = faiss.IndexFlatL2(d)
    # add dataset to index
    index.add(X)
    if use_gpu:
        # try to move to GPU
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    # add dummy self in neighbors query; query k+1 and drop self
    D, I = index.search(X, k)
    return D, I


def faiss_range_counts(X, rs, use_gpu=False):
    """For each query point in X, count number of neighbors within each radius in rs.

    Returns an array of length len(rs) with total number of pairs (i<j) with distance<r.
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError('faiss is not installed')
    X = np.asarray(X).astype('float32')
    n, d = X.shape
    index = faiss.IndexFlatL2(d)
        # add dataset to index
        index.add(X)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass

    counts = []
    # faiss.range_search expects squared radius
    for r in rs:
        r2 = float(r * r)
            lims, D, I = index.range_search(X, r2)
        # lims is of length n+1, number of hits for query i = lims[i+1]-lims[i]
        hits_per_query = np.diff(lims)
        total_hits = int(hits_per_query.sum())
        # range_search returns self hits as well; subtract n (self-self) to count i!=j
        total_pairs = max(0, total_hits - n)
        # each pair counted twice across queries (i->j and j->i), so divide by 2
        counts.append(total_pairs // 2)
    return np.array(counts, dtype=np.int64)
