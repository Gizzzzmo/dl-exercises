import numpy as np
from .Base import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.last_shape = input_tensor.shape
        return input_tensor.reshape(len(input_tensor), np.prod(input_tensor.shape[1:]))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.last_shape)