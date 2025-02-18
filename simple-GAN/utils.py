import torch


def noise_gen(n_samples, n_vector_dim, device='cpu'):
    return torch.randn(n_samples, n_vector_dim, device=device)


def augmentation(mode='train'):
    pass