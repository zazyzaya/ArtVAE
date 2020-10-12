import os
import torch 

from tqdm import tqdm 
from torchvision.transforms import ToTensor
from PIL import Image, UnidentifiedImageError


TRAIN_FILES = os.path.join(os.getcwd(), 'dataset', 'training_set')
TEST_FILES = os.path.join(os.getcwd(), 'dataset', 'validation_set')

class ImgSet():
    def __init__(self, width=256, height=256, gray=False):
        self.width = width
        self.height = height
        self.color = 'LA' if gray else 'RGB'

        self.class_map = dict()
        self.X = None 
        self.y = None 

    '''
    Builds tensors from every image contained in the subfolders of location
    assumes each subfolder is a different label
    '''
    def load_folders(self, location, ignore=[], only_use=None):
        ignore.append('.DS_STORE')
        imgs = []
        y = []

        tt = ToTensor()
        
        for folder in os.listdir(location):
            if folder in self.class_map:
                label = self.class_map[folder]
            else:
                label = len(self.class_map)
                self.class_map[folder] = label 

            if folder.upper() in ignore:
                continue 

            if only_use:
                generator = [os.path.join(location, folder, s) for s in only_use]
            else:
                generator = os.listdir(os.path.join(location, folder))

            print("Loading %s" % folder)
            for img_file in tqdm(generator):
                if img_file.upper() == '.DS_STORE':
                    continue 

                img_file = os.path.join(location, folder, img_file)
                
                # Some images are inexplicably broken. Just skip them, it's
                # faster than manually removing them when it crashes
                try:
                    img = Image.open(img_file)
                except UnidentifiedImageError:
                    continue 

                img = img.resize((self.width, self.height))
                img = img.convert(self.color)

                imgs.append(tt(img))
                y.append(label)

        if self.X == None:
            self.X = torch.stack(imgs)
            self.y = torch.tensor(y)
        else:
            self.X = torch.cat([self.X, torch.stack(imgs)])
            self.y = torch.cat([self.y, torch.tensor(y)])

if __name__ == '__main__':
    s = ImgSet()
    s.load_folders('data')