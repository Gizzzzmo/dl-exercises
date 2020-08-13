import numpy as np
from .Base import Layer

class Sigmoid(Layer):
    def forward(self, input_tensor):
        self.last_sig = 1/(1 + np.exp(-input_tensor))
        return self.last_sig

    def backward(self, error_tensor):
        return self.last_sig * (1 - self.last_sig) * error_tensor
