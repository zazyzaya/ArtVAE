import torch
from matplotlib import pyplot as plt

from random import randint
from data_loader import ImgSet
from vae import VAE
from torch.optim import Adam, Adadelta
from torchvision.transforms import ToPILImage

EPOCHS = 100
DEMOS = 10
LR = 0.001

imgs = ImgSet()
imgs.load_folders('data')
X = imgs.X

model = VAE(imgs.height, imgs.width)
opt = Adadelta(model.parameters())

for i in range(EPOCHS):
    opt.zero_grad()
    X_prime, mu, std = model(X)
    loss = model.loss(X, X_prime, mu, std)

    loss.backward()
    opt.step()

    print("[%d] %0.4f" % (i, loss.item()))

torch.save(model, open('ArtAI.model', 'wb'))

to_img = ToPILImage()
idx = torch.randperm(X.size()[0])[:DEMOS]

for i in idx:
    f, ax_arr = plt.subplots(2)
    ax_arr[0].imshow(to_img(X[i]))
    ax_arr[1].imshow(to_img(model(X[i].unsqueeze(dim=0))[0][0]))
    plt.show()