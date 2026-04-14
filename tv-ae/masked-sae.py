#this code implements the masked AE
import torch as torch
import torch.nn as nn 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Standard AE: (will need to situate this "novel" contribution in literature)
class AE(nn.Module):
    def __init__(self,nambient = 28*28, nlatent = 64, nhidden = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nambient, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nlatent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nlatent, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nambient),
            nn.Sigmoid() #assume data are normalized
        )
        
        #mask (enforce sparsity on w)
        self.w = nn.Parameter(0.01*torch.randn(nlatent))
        
    def forward(self, x):
        z = self.encoder(x)
        #normalzie output
        #z = z/(z.norm(dim=1, keepdim=True))
        #apply mask
        z = z*self.w
        x_hat = self.decoder(z)
        return x_hat, z
    
    
class getData():
    def __init__(self, dataset, noise_level, true_dim):
        super().__init__()
        
        self.noise_level = noise_level
        self.true_dim = true_dim
        
        if dataset == "mnist":
            
            #construct path to load
            #chatGPT fix
            BASE_DIR = Path(__file__).resolve().parent.parent
            datapath = BASE_DIR / "datasets" / f"b{true_dim}_n{noise_level}.npy"
            if not os.path.exists(datapath):
                print(datapath)
                raise FileExistsError("Dataset does not exist")
            
            self.X = np.load(datapath)
            self.X = torch.from_numpy(self.X).float()
            
            N = self.X.shape[0]
            train_size = int(0.8*N)

            #NOTE FOR REPRODUCIBILITY, this should be fixed, and saved beforehand (i.e. read a file of indices)
            indices = torch.randperm(N)

            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

            train_X = self.X[train_idx]
            test_X = self.X[test_idx]

            train_ds = TensorDataset(train_X)
            test_ds = TensorDataset(test_X)

            self.train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            self.test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            self.dataframe = DataLoader(self.X, batch_size=128, shuffle = False)
    
    
##THIS TEST CODE IS GPT GENERATED
# transform = transforms.ToTensor()

# train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
# test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)


#this class will build the dimension estimator and then output elbow (or knee?) diagram
class MAEestimator():
    
    def __init__(self, data, nambient=28*28, nlatent = 64, nhidden = 256, lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2], lr = 1e-3, epochs = 5):

        #initalize quantities that we need:
        self.nambient = nambient
        self.nhidden = nhidden
        self.nlatent = nlatent
        self.lambdas = lambdas
        self.lr = lr
        self.epochs = epochs
        
        self.true_dim = data.true_dim
        self.noise_level = data.noise_level
        
        #initalize train and test-data
        #assuems getdata object:
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        
        #results
        self.results = []
        
        
    def evaluate_full(self, model):
        model.eval()
        total_recon = 0
        
        with torch.no_grad():
            for (x,) in self.test_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                recon = F.mse_loss(x_hat, x.view(x.size(0), -1), reduction='mean')
                total_recon += recon.item()
        
        total_recon /= len(self.test_loader.dataset)
        
        w = model.w.detach().cpu()
        #therhold... this may not be the best way to do this
        active = (w.abs() > 1e-2).sum().item()
        
        return total_recon, active
        
    def sweep_lambdas(self):
        epochs = self.epochs
        for lambda_w in self.lambdas:
            print(f"\n=== λ = {lambda_w} ===")
            
            # reinitialize model each time
            model = AE(nambient = self.nambient, nlatent = self.nlatent).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            # train
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for (x,) in self.train_loader:
                    x = x.to(device)
                    
                    optimizer.zero_grad()
                    x_hat, z = model(x)
                    
                    recon_loss = F.mse_loss(x_hat, x.view(x.size(0), -1), reduction='mean')
                    sparsity_loss = torch.norm(model.w, 1)
                    
                    loss = recon_loss + lambda_w * sparsity_loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}: loss = {total_loss/len(self.train_loader):.4f}")
            
            recon, active = self.evaluate_full(model)
            self.results.append((lambda_w, recon, active))
            print(model.w)
            
            print(f"λ={lambda_w:.5f} | recon={recon:.6f} | active dims={active}")
            
        # unpack
        lams, recons, dims = zip(*self.results)
        
        plt.figure()
        plt.plot(dims, recons, marker='o')
        plt.xlabel("Active latent dimensions")
        plt.ylabel("Reconstruction error")
        plt.title(
            f"Rate–distortion curve\n"
            f"true_dim={self.true_dim}, noise={self.noise_level}"
        )

        plt.savefig(
            f"rate_distortion_d{self.true_dim}_n{self.noise_level}.png",
            bbox_inches="tight"
        )
        plt.show()
        
        

            
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = getData(
        dataset="mnist",
        noise_level=0.2,
        true_dim=20
    )

    estimator = MAEestimator(
        data=data,
        nambient=28*28,
        nlatent=64,
        nhidden=256,
        lambdas=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.65],
        lr=1e-3,
        epochs=5
    )

    estimator.sweep_lambdas()
        
    




# model = AE(nlatent=64).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# lambda_w = 1e-1


# def train_epoch():
#     model.train()
#     total_loss = 0
    
#     for x, _ in train_loader:
#         x = x.to(device)
        
#         optimizer.zero_grad()
        
#         x_hat, z = model(x)
        
#         # reconstruction loss
#         recon_loss = F.mse_loss(x_hat, x.view(x.size(0), -1))
        
#         # sparsity on w
#         sparsity_loss = torch.norm(model.w, 1)
        
#         loss = recon_loss + lambda_w * sparsity_loss
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     return total_loss / len(train_loader)

# # ----------------------
# # Evaluation
# # ----------------------
# def evaluate():
#     model.eval()
#     with torch.no_grad():
#         w = model.w.detach().cpu()
        
#         # count "active" dims
#         active = (w.abs() > 1e-2).sum().item()
        
#         print(f"Active latent dims: {active}/{len(w)}")
#         print(f"w (first 10): {w[:10]}")
        
#         return active, w

# # ----------------------
# # Visualization
# # ----------------------
# def show_reconstructions():
#     model.eval()
#     x, _ = next(iter(test_loader))
#     x = x.to(device)
    
#     with torch.no_grad():
#         x_hat, _ = model(x)
    
#     x = x.cpu()
#     x_hat = x_hat.view(-1, 1, 28, 28).cpu()
    
#     fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    
#     for i in range(8):
#         axes[0, i].imshow(x[i].squeeze(), cmap="gray")
#         axes[0, i].axis("off")
        
#         axes[1, i].imshow(x_hat[i].squeeze(), cmap="gray")
#         axes[1, i].axis("off")
    
#     axes[0, 0].set_title("Original")
#     axes[1, 0].set_title("Recon")
#     plt.show()

# # ----------------------
# # Sweep lambda
# # ----------------------
# lambdas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

# results = []

# def evaluate_full():
#     model.eval()
#     total_recon = 0
    
#     with torch.no_grad():
#         for x, _ in test_loader:
#             x = x.to(device)
#             x_hat, _ = model(x)
#             recon = F.mse_loss(x_hat, x.view(x.size(0), -1), reduction='sum')
#             total_recon += recon.item()
    
#     total_recon /= len(test_loader.dataset)
    
#     w = model.w.detach().cpu()
#     active = (w.abs() > 1e-2).sum().item()
    
#     return total_recon, active

# epochs = 5
# for lambda_w in lambdas:
#     print(f"\n=== λ = {lambda_w} ===")
    
#     # reinitialize model each time
#     model = AE(nlatent=64).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
#     # train
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
        
#         for x, _ in train_loader:
#             x = x.to(device)
            
#             optimizer.zero_grad()
#             x_hat, z = model(x)
            
#             recon_loss = F.mse_loss(x_hat, x.view(x.size(0), -1))
#             sparsity_loss = torch.norm(model.w, 1)
            
#             loss = recon_loss + lambda_w * sparsity_loss
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")
    
#     recon, active = evaluate_full()
#     results.append((lambda_w, recon, active))
    
#     print(f"λ={lambda_w:.5f} | recon={recon:.6f} | active dims={active}")
    
# # unpack
# lams, recons, dims = zip(*results)

# plt.figure()
# plt.plot(dims, recons, marker='o')
# plt.xlabel("Active latent dimensions")
# plt.ylabel("Reconstruction error")
# plt.title("Rate–distortion curve")
# plt.show()
# plt.savefig("plot.png")

# plt.figure()
# plt.plot(lambdas, dims)
# plt.show()
# plt.savefig("dimvlambda.png")