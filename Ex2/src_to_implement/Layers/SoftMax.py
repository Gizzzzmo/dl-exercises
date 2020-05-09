import numpy as np

class SoftMax:
    def __init__(self):
        pass
    
    def forward(self, input_tensor):
        self.last_exp = np.exp(input_tensor - np.max(input_tensor, axis=1).reshape(len(input_tensor), 1))
        self.last_summed = self.last_exp.sum(axis=1).reshape(len(input_tensor), 1)
        return self.last_exp / self.last_summed

    def backward(self, error_tensor):
        amount = len(self.last_exp)
        classes = self.last_exp.shape[1]

        term1 = self.last_exp/(self.last_summed**2)

        diagonal = term1 * (self.last_summed - self.last_exp)
        differential = -term1.reshape(amount, classes, 1) @ self.last_exp.reshape(amount, 1, classes)
        differential.reshape(amount, classes**2)[:, ::classes+1] = diagonal

        return (error_tensor.reshape(amount, 1, classes) @ differential).reshape(amount, classes)
