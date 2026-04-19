import numpy as np
import logging
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


    def _piecewise_breakpoint(xs, ys):
        """Find a single breakpoint by fitting a continuous piecewise-linear
        function with one knot (breakpoint) via least-squares.

        We search candidate breakpoints between unique sorted x-values and
        choose the one that minimizes the residual sum of squares. Returns
        the x-coordinate of the selected breakpoint (float).
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if xs.size == 0:
            return float('nan')

        # sort by xs
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]

        # If too few unique points, return central x
        uniq = np.unique(xs_s)
        if uniq.size < 2 or xs_s.size < 3:
            return float(xs_s[xs_s.size // 2])

        # candidate breakpoints: midpoints between consecutive unique x's
        candidates = (uniq[:-1] + uniq[1:]) / 2.0

        best_bp = float(candidates[0])
        best_rss = np.inf

        for x0 in candidates:
            # design matrix: [1, x, max(0, x-x0)] (continuous piecewise linear)
            H = np.vstack([np.ones_like(xs_s), xs_s, np.maximum(0.0, xs_s - x0)]).T
            # least squares solve
            coef, residuals, rank, s = np.linalg.lstsq(H, ys_s, rcond=None)
            if residuals.size > 0:
                rss = float(residuals[0])
            else:
                pred = H.dot(coef)
                rss = float(((ys_s - pred) ** 2).sum())
            if rss < best_rss:
                best_rss = rss
                best_bp = float(x0)

        return best_bp


    def masked_ae_estimate(X, nlatent=64, nhidden=256, lambdas=None,
                          lr=1e-3, epochs=10, batch_size=128, device=None,
                          threshold=1e-3,
                          pretrain_epochs=50, pretrain_lr=1e-4,
                          sweep_epochs=25, sweep_lr=1e-5,
                          return_debug=False,
                          **kwargs):
        """Estimate intrinsic dimension via masked AE lambda-sweep and 1-breakpiece fit.

        Returns: float (estimated dimension = number of active latents at fitted breakpoint)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if lambdas is None:
            # 10 lambdas spaced log-uniformly between 0.1 and 10 (user spec)
            lambdas = list(np.logspace(np.log10(0.1), np.log10(10.0), 10))

        logger = logging.getLogger(__name__)
        # DEBUG: report lambda sweep we're using
        logger.debug(f"[masked-ae DEBUG] lambda_grid={lambdas} nlatent={nlatent} nhidden={nhidden} epochs={epochs} lr={lr} threshold={threshold}")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        n, D = X.shape
        ds = TensorDataset(torch.from_numpy(X).float())
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        results = []  # list of (lambda, recon, active)

        # Pretrain without sparsity to obtain good recon baseline (warm start)
        logger.debug(f"[masked-ae DEBUG] Pretraining {pretrain_epochs} epochs lr={pretrain_lr} (no sparsity)")
        base_model = AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
        opt_pre = torch.optim.Adam(base_model.parameters(), lr=pretrain_lr)
        for ep in range(pretrain_epochs):
            base_model.train()
            for (batch,) in dl:
                batch = batch.to(device)
                opt_pre.zero_grad()
                x_hat, z = base_model(batch)
                recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                loss = recon
                loss.backward()
                opt_pre.step()

        # save pretrained state
        pretrained_state = base_model.state_dict()

        # Sweep lambdas warm-starting from pretrained weights
        logger.debug(f"[masked-ae DEBUG] Sweeping {len(lambdas)} lambdas warm-starting from pretrained checkpoint; sweep_epochs={sweep_epochs} sweep_lr={sweep_lr}")
        for lam in lambdas:
            # restore model to pretrained state for each lambda
            model = AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
            model.load_state_dict(pretrained_state)
            opt = torch.optim.Adam(model.parameters(), lr=sweep_lr)

            for ep in range(sweep_epochs):
                model.train()
                for (batch,) in dl:
                    batch = batch.to(device)
                    opt.zero_grad()
                    x_hat, z = model(batch)
                    recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                    # penalize the masked activations (mean L1 over batch and latents)
                    spars = torch.mean(torch.abs(z * model.w))
                    loss = recon + lam * spars
                    loss.backward()
                    opt.step()

            # evaluate reconstruction error (mean per-sample MSE)
            model.eval()
            with torch.no_grad():
                total_recon = 0.0
                for (batch,) in dl:
                    batch = batch.to(device)
                    x_hat, _ = model(batch)
                    recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
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
            logger.debug(f"[masked-ae DEBUG] lam={lam} recon_mean_per_sample={total_recon:.6g} active={active} w_min={w_min} w_max={w_max} w_mean={w_mean}")

        # unpack
        lams, recons, acts = zip(*results)
        lams = np.asarray(lams, dtype=float)
        acts = np.asarray(acts)
        recons = np.asarray(recons)

        # DEBUG: report arrays that go into Kneedle
        logger.debug(f"[masked-ae DEBUG] lams={lams.tolist()} acts={acts.tolist()} recons={recons.tolist()}")

        # Use log(lambda) -> active for elbow detection (more sensitive)
        log_lams = np.log10(lams)
        try:
            bp_log = _piecewise_breakpoint(log_lams, acts)
            lam_bp = float(10 ** bp_log)
            logger.debug(f"[masked-ae DEBUG] piecewise_bp_log={bp_log} lam_bp={lam_bp}")
        except Exception as e:
            logger.debug(f"[masked-ae DEBUG] piecewise fit exception: {e}")
            lam_bp = float(lams[int(len(lams) // 2)])


        # Retrain once at the selected lambda (warm-start from pretrained) to get final active count
        logger.debug(f"[masked-ae DEBUG] Retraining at selected lambda={lam_bp} from pretrained checkpoint, epochs={sweep_epochs}, lr={sweep_lr}")
        model = AE(nambient=D, nlatent=nlatent, nhidden=nhidden).to(device)
        # use pretrained_state saved above if available
        try:
            model.load_state_dict(pretrained_state)
        except Exception:
            pass
        opt = torch.optim.Adam(model.parameters(), lr=sweep_lr)
        for ep in range(sweep_epochs):
            model.train()
            for (batch,) in dl:
                batch = batch.to(device)
                opt.zero_grad()
                x_hat, z = model(batch)
                recon = F.mse_loss(x_hat, batch.view(batch.size(0), -1), reduction='mean')
                spars = torch.mean(torch.abs(z * model.w))
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
        logger.debug(f"[masked-ae DEBUG] final lam={lam_bp} active_final={active_final} w_min={w_min} w_max={w_max} w_mean={w_mean}")

        d_hat = float(max(0, min(nlatent, int(active_final))))
        logger.debug(f"[masked-ae DEBUG] d_hat={d_hat}")

        if return_debug:
            # package diagnostics for plotting / inspection
            w_sorted = np.sort(w_final)[::-1]
            meta = {
                'lam_bp': float(lam_bp),
                'bp_log': float(bp_log),
                'active_final': int(active_final),
                'threshold': float(threshold),
                'nlatent': int(nlatent),
                'nhidden': int(nhidden),
                'pretrain_epochs': int(pretrain_epochs),
                'sweep_epochs': int(sweep_epochs),
                'pretrain_lr': float(pretrain_lr),
                'sweep_lr': float(sweep_lr),
            }
            return {
                'd_hat': d_hat,
                'lams': lams,
                'acts': acts,
                'recons': recons,
                'lam_bp': lam_bp,
                'bp_log': bp_log,
                'w_final': w_final,
                'w_sorted': w_sorted,
                'meta': meta,
            }

        return d_hat
