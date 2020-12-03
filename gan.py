import torch 
import torchvision

from torch import nn 
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, hidden_channels=128, out_channels=3):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.out_size = out_size
        self.hc = hidden_channels

        # Div by 4 bc 2 upsamples that double mat size 
        self.resizer = nn.Linear(latent_size, self.hc * (self.out_size//4) ** 2)
    
        # Note: convs w no stride or padding don't change img size 
        self.convs = nn.Sequential(
            nn.BatchNorm2d(self.hc),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hc, self.hc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hc, self.hc//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hc//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hc//2, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    '''
    Takes 1D noise vectors and converts to n-channel image tensors w same
    size as output size
    '''
    def resize_noise(self, z):
        x = self.resizer(z)
        x = x.view(x.size()[0], self.hc, self.out_size//4, self.out_size//4)
        return x

    def forward(self, z):
        x = self.resize_noise(z)
        x = self.convs(x)
        return x 

    def generate_random(self, batch_size):
        with torch.no_grad():
            z = get_noise(self.latent_size, batch_size)
            x = self.forward(z)

        return x.detach()

class Discriminator(nn.Module):
    def __init__(self, img_size, in_channels=3, out_channels=128):
        super(Discriminator, self).__init__()

        # Generator for downsampling conv that decreases size by 1/2
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # I'm not typing that damn thing every time
        oc = out_channels

        self.model = nn.Sequential(
            *discriminator_block(in_channels, oc//2**3, bn=False),
            *discriminator_block(oc//2**3, oc//2**2),
            *discriminator_block(oc//2**2, oc//2),
            *discriminator_block(oc//2, oc)
        )

        # Halved 4 times
        new_size = img_size // 2 ** 4
        self.decider = nn.Sequential(
            nn.Linear(oc * new_size ** 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size()[0], -1)
        return self.decider(out)

def get_noise(dim, batch_size):
    return Variable(
        torch.empty((batch_size, dim)).normal_(mean=0, std=1)
    )

g_loss = nn.BCELoss()
d_loss = nn.MSELoss()