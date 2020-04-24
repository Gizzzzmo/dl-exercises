import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        if (resolution % (2*tile_size) != 0):
            raise Exception('resolution must be evenly divisible by two times the tile_size')
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.empty((resolution, resolution))

    def draw(self):
        stripes = np.tile(np.concatenate((np.zeros(self.tile_size, dtype=int), np.ones(self.tile_size, dtype=int))), (self.resolution, self.resolution//(2*self.tile_size)))
        output = stripes ^ stripes.transpose()
        self.output[:] = output
        return output

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = np.array(position)
        self.output = np.empty((resolution, resolution))

    def draw(self):
        x = np.tile(np.arange(0, self.resolution), (self.resolution, 1))
        xy = np.stack((x, x.transpose()), axis=-1)
        output = (np.sum(np.square(xy - self.position), axis=-1) <= self.radius ** 2) * 1
        self.output[:] = output
        return output
