import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
if not TORCH_AVAILABLE:
    def masked_ae_estimate(*args, **kwargs):
        raise RuntimeError("PyTorch is required for masked AE estimator")
else:
    class AE(nn.Module):
        def __init__(self, nambient, nlatent=64, nhidden=256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nambient, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nlatent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(nlatent, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nambient),
                nn.Sigmoid(),
            )
            # learnable per-latent weight (the "mask")
            self.w = nn.Parameter(0.01 * torch.randn(nlatent))

        def forward(self, x):
            z = self.encoder(x)
            z = z * self.w
            x_hat = self.decoder(z)
            return x_hat, z


    def _kneedle(xs, ys):
        """Detect elbow using a Kneedle-like procedure.

        This implementation normalizes `xs` and `ys` to [0,1], computes the
        difference curve and selects the index of maximal deviation as the
        elbow. Returns the x-coordinate at the detected elbow (float).
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if xs.size == 0:
            return float('nan')

        # sort by xs
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]

        # If there are too few unique points, return the median
        if np.unique(xs_s).size < 2:
            return float(xs_s[0])

        # normalize to [0,1]
        def _norm(a):
            a_min = a.min()
            a_max = a.max()
            if a_max <= a_min:
                return np.zeros_like(a)
            return (a - a_min) / (a_max - a_min)

        x_n = _norm(xs_s)
        y_n = _norm(ys_s)

        # determine monotonicity of y: if y decreases with x -> decreasing=True
        decreasing = y_n[0] > y_n[-1]

        # Kneedle score: difference between the curve and the diagonal.
        # For decreasing curves we want x - y, otherwise y - x.
        if decreasing:
            score = x_n - y_n
        else:
            score = y_n - x_n

        # small smoothing to reduce noise (moving average window=3)
        if score.size >= 3:
            kernel = np.ones(3) / 3.0
            score_smooth = np.convolve(score, kernel, mode='same')
        else:
            score_smooth = score

        # elbow index: location of maximal score
        idx = int(np.argmax(score_smooth))
        return float(xs_s[idx])


    def masked_ae_estimate(X, nlatent=64, nhidden=256, lambdas=None, lr=1e-3, epochs=5, batch_size=128, device=None, threshold=1e-3, **kwargs):
        """Estimate intrinsic dimension via masked AE lambda-sweep and 1-breakpiece fit.

        Returns: float (estimated dimension = number of active latents at fitted breakpoint)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if lambdas is None:
            # wider lambda sweep (log-spaced-ish) to ensure we reach strong sparsity
            lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0]

        # DEBUG: report lambda sweep we're using
        print(f"[masked-ae DEBUG] lambda_grid={lambdas} nlatent={nlatent} nhidden={nhidden} epochs={epochs} lr={lr} threshold={threshold}")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        n, D = X.shape
        ds = TensorDataset(torch.from_numpy(X).float())
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        results = []  # list of (lambda, recon, active)

        for lam in lambdas:
            model = AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            for ep in range(epochs):
                model.train()
                total_loss = 0.0
                for (batch,) in dl:
                    batch = batch.to(device)
                    opt.zero_grad()
                    x_hat, z = model(batch)
                    recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                    spars = torch.norm(model.w, 1)
                    loss = recon + lam * spars
                    loss.backward()
                    opt.step()
                    total_loss += float(loss.item()) * batch.size(0)

            # evaluate reconstruction error (mean per-sample MSE)
            model.eval()
            with torch.no_grad():
                total_recon = 0.0
                for (batch,) in dl:
                    batch = batch.to(device)
                    x_hat, _ = model(batch)
                    recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='sum')
                    total_recon += float(recon.item())
            total_recon /= float(n)

            w = model.w.detach().cpu().abs().numpy()
            active = int((w > threshold).sum())
            results.append((lam, total_recon, active))
            # DEBUG: log per-lambda results and some mask stats
            try:
                w_min, w_max, w_mean = float(w.min()), float(w.max()), float(w.mean())
            except Exception:
                w_min = w_max = w_mean = None
            print(f"[masked-ae DEBUG] lam={lam} recon_mean_per_sample={total_recon:.6g} active={active} w_min={w_min} w_max={w_max} w_mean={w_mean}")

        # unpack
        lams, recons, acts = zip(*results)
        lams = np.asarray(lams, dtype=float)
        acts = np.asarray(acts)
        recons = np.asarray(recons)

        # DEBUG: report arrays that go into Kneedle
        print(f"[masked-ae DEBUG] lams={lams.tolist()} acts={acts.tolist()} recons={recons.tolist()}")

        # Use log(lambda) -> active for elbow detection (more sensitive)
        log_lams = np.log10(lams)
        try:
            bp_log = _kneedle(log_lams, acts)
            lam_bp = float(10 ** bp_log)
            print(f"[masked-ae DEBUG] kneedle_bp_log={bp_log} lam_bp={lam_bp}")
        except Exception as e:
            print(f"[masked-ae DEBUG] kneedle exception: {e}")
            lam_bp = float(lams[int(len(lams) // 2)])

        # Retrain once at the selected lambda (possibly interpolated) to get final active count
        print(f"[masked-ae DEBUG] Retraining at selected lambda={lam_bp}")
        model = AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for (batch,) in dl:
                batch = batch.to(device)
                opt.zero_grad()
                x_hat, z = model(batch)
                recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                spars = torch.norm(model.w, 1)
                loss = recon + lam_bp * spars
                loss.backward()
                opt.step()

        # evaluate final active latents
        model.eval()
        with torch.no_grad():
            w_final = model.w.detach().cpu().abs().numpy()
        active_final = int((w_final > threshold).sum())
        try:
            w_min, w_max, w_mean = float(w_final.min()), float(w_final.max()), float(w_final.mean())
        except Exception:
            w_min = w_max = w_mean = None
        print(f"[masked-ae DEBUG] final lam={lam_bp} active_final={active_final} w_min={w_min} w_max={w_max} w_mean={w_mean}")

        d_hat = float(max(0, min(nlatent, int(active_final))))
        print(f"[masked-ae DEBUG] d_hat={d_hat}")
        return d_hat
