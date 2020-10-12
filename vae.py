import torch

from torch.nn import Linear, Sigmoid, ReLU, Conv2d, \
    ConvTranspose2d, AdaptiveAvgPool2d, ModuleList, \
    Sequential, BCELoss

class VAE(torch.nn.Module):
    def __init__(self, input_x, input_y, n_convs=4, ksize=3, padding=1, stride=2, 
                embedding_dim=16, colors=3):
        super().__init__()

        self.n_convs = n_convs
        self.colors = colors

        enc = []
        for i in range(n_convs):
            enc += [
                Conv2d(colors*2**i, colors*2**(i+1), ksize+i, stride=stride, padding=padding),
                ReLU()
            ]
        enc = ModuleList(enc)
        self.enc = Sequential(*enc)

        #self.mean = AdaptiveAvgPool2d(embedding_dim)
        #self.std = AdaptiveAvgPool2d(embedding_dim)

        dec = []
        for i in range(n_convs, 0, -1):
            dec += [
                ConvTranspose2d(colors*2**i, colors*2**(i-1), ksize+(i), stride=stride, padding=padding),
                ReLU()
            ]
        
        dec += [AdaptiveAvgPool2d((input_x, input_y))]

        dec = ModuleList(dec)
        self.dec = Sequential(*dec)

        self.criterion = BCELoss(reduction='sum')

    def forward(self, X):
        x = self.enc(X)
        print(x.size())
        
        #mu = self.mean(x)
        #std = self.std(x)

        mu = x
        std = x 
        z = self.reparam(mu, std)
        print(z.size())

        x = torch.sigmoid(self.dec(z))
        print(x.size())

        return x, mu, std

    def reparam(self, mu, std):
        std = torch.exp(0.5*std) 
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def loss(self, X, X_prime, mu, std):
        kl_loss = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp())
        bce = self.criterion(X_prime, X)

        return bce+kl_loss

    '''
    Create noise embeddings and see what they decode to 
    '''
    def generate(self, samples=10, size=14):
        with torch.no_grad():
            z = torch.rand((
                samples, 
                self.colors*2**(self.n_convs),
                size,
                size
            ))

            return torch.sigmoid(self.dec(z))