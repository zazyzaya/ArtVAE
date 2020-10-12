import torch 
import sys 

from data_loader import ImgSet
from vae import VAE
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

SIZE = 256 * 8
DEMOS = 10
s = ImgSet(SIZE, SIZE, gray=True, max_files=25)
s.load_folders('data')
X = s.X

if len(sys.argv) <= 1:
    fname = 'ArtAI.model'
else:
    fname = sys.argv[1]

model = torch.load(open(fname, 'rb'))

to_img = ToPILImage()
idx = torch.randperm(X.size()[0])[:DEMOS]

for i in idx:
    f, ax_arr = plt.subplots(2)
    ax_arr[0].imshow(to_img(X[i]))
    ax_arr[1].imshow(to_img(model(X[i].unsqueeze(dim=0))[0][0]))
    plt.show()