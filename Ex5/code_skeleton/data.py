from torch.utils.data import Dataset
import torch
import random
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import itertools

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]



class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode
        self._transform_test = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    @property
    def transform(self):
        if self.mode == 'train':
            return self._transform_train
        else:
            return self._transform_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if type(index) is slice:
            imgs = []
            labels = []
            for i in range(index.start, index.stop, index.step):
                img_path, crack, inactive = self.data.iloc[i][0:3]
                imgs.append(self.transform(gray2rgb(imread(img_path))))
                labels.append(torch.Tensor([int(crack), int(inactive)]))

            return torch.stack(imgs), torch.stack(labels)
        elif type(index) is int:
            img_path, crack, inactive = self.data.iloc[index][0:3]
            img = self.transform(gray2rgb(imread(img_path)))
            label = torch.Tensor([int(crack), int(inactive)])

            return img, label
        else:
            raise KeyError()



class DataLoader(object):
    def __init__(self, x, y, batch_size, shuffle=False, f=lambda a:a, augment=False):
        if shuffle:
            perm = torch.randperm(len(x))
            self.x = x[perm]
            self.y = y[perm]
        else:
            self.x = x
            self.y = y
        self.shuffle = shuffle

        if augment:
            def augment(inn):
                rots = random.choice([0, 1, 2, 3])
                for i in range(rots):
                    inn = inn.transpose(2, 3)
                    inn = inn.flip(dims=(2,))
                if random.choice([True, False]):
                    inn = inn.flip(dims=(2,))
            
                return inn
        else:
            augment = lambda a:a
        gen = ((f(augment(self.x[i:i+batch_size])), self.y[i:i+batch_size]) for i in range(0, len(x), batch_size))

        self.backup, self.gen = itertools.tee(gen)
        self.length = (len(x) + batch_size - 1)//batch_size

    def reset(self):
        self.backup, self.gen = itertools.tee(self.backup)
        if self.shuffle:
            perm = torch.randperm(len(self.x))
            self.x = self.x[perm]
            self.y = self.y[perm]

    def __len__(self): 
        return self.length

    def __next__(self):
        try:
            nx = next(self.gen)
            return nx
        except StopIteration:
            self.reset()
            raise StopIteration

    def __iter__(self):
        return self