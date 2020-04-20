import os.path
import json
import glob
import random
import cv2
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation
        with open(label_path) as label_file:
            self.labels = json.load(label_file)
        self.files = glob.glob(file_path + '/*')

        self.currentfile = 0

        if shuffle:
            random.shuffle(self.files)
        

    def next(self):
        start = self.currentfile
        end = min(self.currentfile + self.batch_size, len(self.files))
        
        files = self.files[start:end]
        
        if (end == len(self.files)):
            # assuming that the batch size is smaller than the size of the dataset
            files += self.files[0:self.batch_size - end + start]
            # restarting iterator
            self.currentfile = 0
            random.shuffle(self.files)
        else:
            self.currentfile += self.batch_size
        images = [self.augment(cv2.resize(np.load(file), dsize=tuple(self.image_size[0:2]))) for file in files]
        labels = np.array([self.labels[file.split('/')[-1].split('.')[0]] for file in files])
        
        batch = np.stack(images)

        return batch, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring:
            mirror = random.choice([0, 1, 2])
            if (mirror == 1):
                img = img[::-1, :]
            elif (mirror == 2):
                img = img[:, ::-1]

        if self.rotation:
            rotations = random.choice([0, 1, 2, 3])
            for _ in range(rotations):
                img = np.swapaxes(img, 0, 1)[:, ::-1]
        
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        batch, labels = self.next()
        rows = (self.batch_size+2)//3

        _, axs = plt.subplots(rows, 3)
        for i, (img, label) in enumerate(zip(batch, labels)):
            axs[i//3, i%3].axis('off')
            axs[i//3, i%3].imshow(img)
            axs[i//3, i%3].set_title(self.class_name(label))
        plt.show()


