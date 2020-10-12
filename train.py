import torch
import sys 

from matplotlib import pyplot as plt
from random import randint
from data_loader import ImgSet
from vae import VAE
from torch.optim import Adam, Adadelta, SGD
from torchvision.transforms import ToPILImage, ToTensor, \
    Compose, ColorJitter, RandomRotation

EPOCHS = 1000
DEMOS = 10
LR = 0.0001
SIZE = 256
MINI_BATCH = 256
GRAY = False

to_img = ToPILImage()
to_ten = ToTensor()

imgs = ImgSet(SIZE, SIZE, gray=GRAY)
imgs.load_folders('data', ignore=['engraving', 'abstract'])
X = imgs.X

model = VAE(SIZE, SIZE, colors=3)
opt = Adam(model.parameters(), lr=LR)

rand_transforms = Compose([
    ColorJitter(hue=0.5),
    RandomRotation(30)
])

'''
There has to be a better way than this...
'''
def get_transformed_input(x, tr):
    transformed = []
    for img in x:
        img = to_img(img)
        img = tr(img)
        transformed.append(to_ten(img))

    return torch.stack(transformed)

for i in range(EPOCHS):
    opt.zero_grad()
    batch = X[torch.randperm(X.size()[0])][:MINI_BATCH]
    X_test = get_transformed_input(batch, rand_transforms)
    X_prime, mu, std = model(X_test)
    loss = model.loss(X_test, X_prime, mu, std)

    loss.backward()
    opt.step()

    print("[%d] %0.4f" % (i, loss.item()))

torch.save(model, open('ArtAI.model', 'wb'))

if len(sys.argv) > 1 and sys.argv[1].upper() in '--DISPLAY':
    idx = torch.randperm(X.size()[0])[:DEMOS]

    for i in idx:
        f, ax_arr = plt.subplots(2)
        ax_arr[0].imshow(to_img(X[i]))
        ax_arr[1].imshow(to_img(model(X[i].unsqueeze(dim=0))[0][0]))
        plt.show()