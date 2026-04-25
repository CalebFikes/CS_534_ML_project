import numpy as np

#sample points in Rd unit sphere
def sample_sphere(d, n, random_state=None):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n, d + 1))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)

#simple d-dimensional torus: sample angles and map to 2d space per circle
#implemented using LLM
def sample_torus(d, n, random_state=None):
    rng = np.random.default_rng(random_state)

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

#this method embeds the data set X in a D dimensional space using a linear transform
#Here, a random matrix A is generated, and Q is used to generate an orthornormal basis for this space, which is applied to the data
def embed_via_random_orthonormal(X, D, random_state=None):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if D <= d:
        return X[:, :D]
    A = rng.normal(size=(D, d))
    Q, _ = np.linalg.qr(A)
    return X @ Q.T

#function to add noise to dataset
def add_noise(X, sigma, random_state=None):
    if sigma is None:
        return X
    rng = np.random.default_rng(random_state)
    noise = rng.normal(scale=sigma, size=X.shape)
    return X + noise
