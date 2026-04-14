import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class SimpleAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec


def load_mnist_npz(path):
    try:
        data = np.load(path)
        if 'train_images' in data:
            imgs = data['train_images']
        elif 'x_train' in data:
            imgs = data['x_train']
        else:
            # try common key names
            keys = list(data.keys())
            imgs = data[keys[0]]
        imgs = imgs.astype(np.float32) / 255.0
        if imgs.ndim == 3:
            # (N, H, W) -> (N, H*W)
            imgs = imgs.reshape(imgs.shape[0], -1)
        elif imgs.ndim == 4:
            imgs = imgs.reshape(imgs.shape[0], -1)
        return imgs
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=400)
    parser.add_argument('--bottleneck', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save-model', required=True)
    parser.add_argument('--save-latents', required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-loss', default='')
    parser.add_argument('--subset-size', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    #Michael: added argument for the noise level
    parser.add_argument('--noise-levels', type = float, default = 0)
    #where we save the dim-constrained recons:
    parser.add_argument("--save_dataset", type = str, default = "datasets")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # debug: report initial configuration
    print(f"[AE START] seed={args.seed} bottleneck={args.bottleneck} epochs={args.epochs} subset={args.subset_size}", flush=True)

    # Locate MNIST-ish data
    cand = [
        os.path.join(args.data_dir, 'mnist.npz'),
        os.path.join(args.data_dir, 'mnist_complex.npz'),
        os.path.join('dataset', 'mnist.npz'),
        os.path.join('dataset', 'mnist_complex.npz'),
    ]

    X = None
    for p in cand:
        if os.path.exists(p):
            X = load_mnist_npz(p)
            if X is not None:
                break

    if X is None:
        #print("[INFO] No local dataset found. Downloading MNIST via torchvision...", flush=True)

        transform = transforms.ToTensor()

        train_ds = datasets.MNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform
        )

        X = train_ds.data.numpy().astype(np.float32) / 255.0
        X = X.reshape(X.shape[0], -1)  # flatten to (N, 784)
    
    if args.subset_size and args.subset_size > 0 and args.subset_size < X.shape[0]:
        X = X[: args.subset_size]
    
    #print("NOISE DEBUG 2")
    #print(args.noise_levels)
    
    if args.noise_levels > 0:
        #chatGPT debug line:
        noise = args.noise_levels*np.random.randn(*X.shape).astype(np.float32)
        X = X + noise
        X = np.clip(X, 0.0, 1.0) #don't exceed 0 or 1

    input_dim = X.shape[1]
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    model = SimpleAE(input_dim=input_dim, hidden_dim=args.hidden_dim, bottleneck=args.bottleneck)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    #this looks good to me, but this is not a denoising AE--I think that is what we want in this case
    model.train()
    import time as _time
    t_start = _time.time()
    losses = []
    for ep in range(args.epochs):
        epoch_loss = 0.0
        t0 = _time.time()
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * batch.size(0)
        epoch_loss /= len(dl.dataset)
        elapsed = _time.time() - t0
        losses.append(epoch_loss)
        print(f"[AE EPOCH] epoch={ep+1}/{args.epochs} loss={epoch_loss:.6f} epoch_time={elapsed:.2f}s device={device}", flush=True)
    total_t = _time.time() - t_start
    print(f"[AE TRAIN COMPLETE] total_time={total_t:.2f}s", flush=True)
    # save model
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save_model)

    # compute latents for all examples
    model.eval()
    with torch.no_grad():
        Z = []
        for (batch,) in DataLoader(ds, batch_size=256, num_workers=0):
            batch = batch.to(device)
            z = model.encoder(batch)
            Z.append(z.cpu().numpy())
        Z = np.vstack(Z)
    os.makedirs(os.path.dirname(args.save_latents) or '.', exist_ok=True)
    np.save(args.save_latents, Z)
    
    #save the reconstructions--what we want since this will give us the dimension-reconstructed examples:
    with torch.no_grad():
        X_recon = []
        for (batch,) in DataLoader(ds, batch_size=256):
            batch = batch.to(device)
            z = model.encoder(batch)
            x_hat = model.decoder(z)
            X_recon.append(x_hat.cpu().numpy())
        X_recon = np.concatenate(X_recon, axis=0)
        
    #Michael: ChatGPT helped write this save
    base_dir = os.path.dirname(args.save_dataset) or '.'
    base_name = os.path.splitext(os.path.basename(args.save_dataset))[0]

    save_name = f"{base_name}/b{args.bottleneck}_n{args.noise_levels}.npy"
    save_path = os.path.join(base_dir, save_name)

    os.makedirs(base_dir, exist_ok=True)
    np.save(save_path, X_recon)
    
    # save per-epoch losses if requested
    if args.save_loss:
        try:
            os.makedirs(os.path.dirname(args.save_loss) or '.', exist_ok=True)
            np.save(args.save_loss, np.array(losses))
            print(f"[AE LOSS SAVED] {args.save_loss}", flush=True)
            # also print final loss for easier parsing
            if len(losses) > 0:
                print(f"[AE FINAL LOSS] {losses[-1]:.6f}", flush=True)
        except Exception as e:
            print(f"[AE LOSS SAVE ERROR] {e}", flush=True)


if __name__ == '__main__':
    main()
