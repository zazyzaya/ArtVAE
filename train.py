import torch
import sys 

from matplotlib import pyplot as plt
from random import randint
from data_loader import ImgSet
from vae import VAE
from stolen_vae import StolenVAE
from torch.optim import Adam, Adadelta, SGD
from torchvision.transforms import ToPILImage, ToTensor, \
    Compose, ColorJitter, RandomRotation

EPOCHS = 50
DEMOS = 10
LR = 0.01
SIZE = 100
MINI_BATCH = 128
GRAY = False

to_img = ToPILImage()
to_ten = ToTensor()

imgs = ImgSet(SIZE, SIZE, gray=GRAY)
imgs.load_folders('data', ignore=['engraving', 'abstract'])
X = imgs.X

#model = VAE(SIZE, SIZE, colors=3, n_convs=4, ksize=4, stride=1)
model = StolenVAE(input_size=SIZE, latent_size=512)

#opt = Adam(model.parameters(), lr=LR)
opt = Adadelta(model.parameters())

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
    mb = 0
    order = torch.randperm(X.size()[0])
    tot_loss = 0 
    opt.zero_grad()

    while mb*MINI_BATCH < X.size()[0]:
        batch = X[order][MINI_BATCH*mb:min(MINI_BATCH*(mb+1), X.size()[0])]
        X_test = batch #get_transformed_input(batch, rand_transforms)
        X_prime, mu, std = model(X_test)
        loss = model.loss(X_test, X_prime, mu, std)

        loss.backward()
        tot_loss += loss.item()
        mb += 1
        print('.', end='')
    
    opt.step()
    print("[%d] %0.4f" % (i, tot_loss/X.size()[0]))

torch.save(model, open('ArtAI.model', 'wb'))

if len(sys.argv) > 1 and sys.argv[1].upper() in '--DISPLAY':
    idx = torch.randperm(X.size()[0])[:DEMOS]

    for i in idx:
        f, ax_arr = plt.subplots(2)
        ax_arr[0].imshow(to_img(X[i]))
        ax_arr[1].imshow(to_img(model(X[i].unsqueeze(dim=0))[0][0]))
        plt.show()