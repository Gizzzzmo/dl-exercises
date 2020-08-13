from .Base import Layer, Phase
import numpy as np

class Dropout(Layer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.phase == Phase.train:
            self.last_mask = np.random.rand(*input_tensor.shape) < self.probability
            return (input_tensor * self.last_mask)/self.probability
        else:
            return input_tensor

    def backward(self, error_tensor):
        return self.last_mask * error_tensor /self.probability
        
