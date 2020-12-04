import torch
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage

to_img = ToPILImage()
G = torch.load('generator.model')

for i in range(5):
    plt.imshow(to_img(G.generate_random(2)[0]))
    plt.show()