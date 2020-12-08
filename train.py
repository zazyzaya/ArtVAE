import torch
from gan import * 

import sys 
from matplotlib import pyplot as plt
from random import randint
from data_loader import ImgSet
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor, \
    Compose, ColorJitter, RandomRotation

EPOCHS = 500
DEMOS = 10
LR = 0.0002
SIZE = 64 
MINI_BATCH = 512
LATENT_SIZE = 16 
HIDDEN = 256 
GRAY = False
K = 1

MAX_FILES=256
I_PITY_THE_FOOL = 0.4

torch.set_num_threads(16)

to_img = ToPILImage()
to_ten = ToTensor()

imgs = ImgSet(SIZE, SIZE, gray=GRAY, max_files=MAX_FILES)
imgs.load_folders('data', ignore=['iconography', 'engraving', 'abstract'])
X = imgs.X

D = Discriminator(SIZE, out_channels=HIDDEN)
G = Generator(LATENT_SIZE, SIZE, hidden_channels=HIDDEN)
d_opt = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
g_opt = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))

rand_transforms = Compose([
    #ColorJitter(hue=0.15),
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

start_foolin = False
ticks = 1
i = 0
while i < EPOCHS:
    mb = 0
    order = torch.randperm(X.size()[0])
    tot_loss = 0 
    
    while mb*MINI_BATCH < X.size()[0]:
        batch = X[order][MINI_BATCH*mb:min(MINI_BATCH*(mb+1), X.size()[0])]
        bs = batch.size()[0]

        static = G.generate_random(bs)

        real_labels = Variable(torch.full((bs,1), 1.0))
        fake_labels = Variable(torch.zeros((bs,1)).float())

        # First train discriminator
        d_opt.zero_grad()

        real_loss = d_loss(D(batch), real_labels)
        fake_loss = d_loss(D(static), fake_labels)

        d_tot_loss = real_loss.mean().item() + fake_loss.mean().item()
        
        real_loss.backward()
        fake_loss.backward()
        d_opt.step()

        # Don't train generator until descriminator is 
        # pretty good at its job
        if d_tot_loss < I_PITY_THE_FOOL:
            start_foolin = True

        # Then train generator every k steps
        if ticks % K == 0 and start_foolin:
            g_opt.zero_grad()

            z = get_noise(LATENT_SIZE, bs)
            imgs = G(z)
            g_tot_loss = g_loss(D(imgs), real_labels)

            g_tot_loss.backward() 
            g_tot_loss = g_tot_loss.item()
            g_opt.step()

        else:
            g_tot_loss = float('nan')

        print('[%d-%d] D Loss: %0.4f \tG Loss %0.4f' % (i, mb, d_tot_loss, g_tot_loss))

        mb += 1
        ticks += 1

    if i % 10 == 0 and i > 0:
        torch.save(G, 'generator.model')

    if start_foolin:
        i += 1
    

torch.save(G, 'generator.model')
torch.save(D, 'descriminator.model')

if len(sys.argv) > 1 and sys.argv[1].upper() in '--DISPLAY':
    idx = torch.randperm(X.size()[0])[:DEMOS]

    for i in idx:
        plt.imshow(to_img(G.generate_random(2)[0]))
        plt.show()
