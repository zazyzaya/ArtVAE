import torch 
import sys 

from data_loader import ImgSet
from vae import VAE
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

SIZE = 100
DEMOS = 10
s = ImgSet(SIZE, SIZE, gray=False, max_files=25)
s.load_folders('data', ignore=['engraving', 'abstract'])
X = s.X

if len(sys.argv) <= 1:
    fname = 'ArtAI.model'
else:
    fname = sys.argv[1]

model = torch.load(open(fname, 'rb'))

to_img = ToPILImage()
idx = torch.randperm(X.size()[0])[:DEMOS]

with torch.no_grad():
    imgs = model(X[idx])[0]

for i in idx:
    f, ax_arr = plt.subplots(2)
    ax_arr[0].imshow(to_img(X[i]))
    ax_arr[1].imshow(to_img(imgs[i]).convert('RGB'))
    plt.show()