import torch 
import sys 

from torch import distributions
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

NUM_SAMPLES = 10
Z_SIZE = 100
N_CONVS = 4 # Saved as model param but i forgot to make it accessable
COLORS = 3

if len(sys.argv) <= 1:
    fname = 'ArtAI.model'
else:
    fname = sys.argv[1]

model = torch.load(open(fname, 'rb'))

rand_size = lambda : torch.rand(
            (
                NUM_SAMPLES, 
                COLORS*2**(N_CONVS),
                Z_SIZE,
                Z_SIZE
            )
        )

with torch.no_grad():
    r = distributions.Poisson(
        rand_size()**0.25
    )
    z = r.sample()
    #z = model.reparam(z,z)

    X = torch.sigmoid(model.dec(z))
    to_img = ToPILImage()

for i in range(NUM_SAMPLES):
    plt.imshow(to_img(X[i][0]).convert('RGB'))
    plt.show()