import os
import numpy as np
import torch
from src.models.autoencoder import Autoencoder

def test_autoencoder_forward():
    model = Autoencoder(input_dim=28*28, hidden_dim=64, bottleneck=16)
    x = torch.randn(4, 28*28)
    xr = model(x)
    assert xr.shape == x.shape

def test_autoencoder_encode_shape():
    model = Autoencoder(input_dim=28*28, hidden_dim=64, bottleneck=16)
    x = torch.randn(8, 28*28)
    z = model.encode(x)
    assert z.shape == (8, 16)
