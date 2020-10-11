import torch

from torch.nn import Linear, Sigmoid, ReLU, Conv2d, ConvTranspose2d, AdaptiveAvgPool2d, Sequential, BCELoss

class VAE(torch.nn.Module):
    def __init__(self, n_convs=5, ksize=4, padding=1, stride=1, 
                embedding_dim=64, colors=3):
        self.enc = Sequential([
            Conv2d(colors, colors, ksize, stride=stride, padding=padding), 
            ReLU()
        ]*(n_convs-1) + [
            Conv2d(colors, 1, ksize, stride=stride, padding=padding)
        ])

        self.mean = AdaptiveAvgPool2d(embedding_dim)
        self.std = AdaptiveAvgPool2d(embedding_dim)

        self.dec = Sequential([
            ConvTranspose2d(1, colors, ksize, stride=stride, padding=padding),
            ReLU()
        ] + [
            ConvTranspose2d(colors, colors, ksize, stride=stride, padding=padding),
            ReLU()
        ]*(n_convs-1))

        self.criterion = BCELoss(reduction='sum')

    def forward(self, X):
        x = self.enc(X)

        mu = self.mean(x)
        std = self.std(x)

        z = self.reparam(mu, std)

        return torch.sigmoid(self.dec(z)), mu, std

    def reparam(self, mu, std):
        std = torch.exp(0.5*std) 
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def loss(self, X, X_prime, mu, std):
        kl_loss = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp())
        bce = self.criterion(X_prime, X)

        return bce+kl_loss

    def generate(self, X, stoch=True):
        with torch.no_grad():
            x = self.enc(X)

            mu = self.mean(x)
            
            if stoch:
                std = self.std(x)
                z = self.reparam(mu, std)
            else:
                z = mu

            return torch.sigmoid(self.dec(z))