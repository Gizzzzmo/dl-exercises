import numpy as np
from .Base import Layer

class TanH(Layer):
    def forward(self, input_tensor):
        self.last_tanh = np.tanh(input_tensor)
        return self.last_tanh

    def backward(self, error_tensor):
        return (1 - self.last_tanh**2) * error_tensor
