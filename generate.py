import os 
import torch
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

modelf = os.path.join('models', 'gen_c0_128in128out.model')

to_img = ToPILImage()
G = torch.load(modelf)

for i in range(5):
    plt.imshow(to_img(G.generate_random(2)[0]))
    plt.show()