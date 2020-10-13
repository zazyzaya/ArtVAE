import torch

from torch import nn 
from vae import VAE
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD

'''
Just stealing this VAE
https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd
'''
class StolenVAE(nn.Module):
    def __init__(self, input_size=64, latent_size=2048):
        super().__init__()

        self.input_size = input_size
        self.conv_size = input_size // 4

        # Also stole their loss fn 
        self.loss_obj = Loss()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(self.conv_size * self.conv_size * 16, latent_size)
        self.fc_bn1 = nn.BatchNorm1d(latent_size)
        self.fc21 = nn.Linear(latent_size, latent_size)
        self.fc22 = nn.Linear(latent_size, latent_size)
        
        # Sampling vector
        self.fc3 = nn.Linear(latent_size, latent_size)
        self.fc_bn3 = nn.BatchNorm1d(latent_size)
        self.fc4 = nn.Linear(latent_size, self.conv_size * self.conv_size * 16)
        self.fc_bn4 = nn.BatchNorm1d(self.conv_size * self.conv_size * 16)
        
        # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encode
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv4 = conv4.view(-1, self.conv_size*self.conv_size* 16)

        # Latent vars
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu = self.fc21(fc1)
        logvar = self.fc22(fc1)

        # Reparam
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)

        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, self.conv_size, self.conv_size)
        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        out = self.conv8(conv7).view(-1, 3, self.input_size, self.input_size)

        return out, mu, logvar

    def loss(self, X, X_prime, mu, std):
        return self.loss_obj(X_prime, X, mu, std)

    '''
    Useful for that neat vector interpolation thingy 
    '''
    def get_latent(self, X):
        with torch.no_grad():
            # Encode
            conv1 = self.relu(self.bn1(self.conv1(x)))
            conv2 = self.relu(self.bn2(self.conv2(conv1)))
            conv3 = self.relu(self.bn3(self.conv3(conv2)))
            conv4 = self.relu(self.bn4(self.conv4(conv3)))
            conv4 = conv4.view(-1, self.conv_size*self.conv_size* 16)

            # Latent vars
            fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
            mu = self.fc21(fc1)
            logvar = self.fc22(fc1)

            # Reparam
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)

            return z