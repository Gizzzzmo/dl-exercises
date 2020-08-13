import numpy as np

class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(weights_shape) * self.constant

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(*weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/(fan_out + fan_in)), weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/fan_in), weights_shape)