import numpy as np

class SoftMax:
    def __init__(self):
        pass
    
    def forward(self, input_tensor):
        self.last_actual_sum = np.exp(input_tensor).sum(axis=1).reshape(len(input_tensor), 1, 1)
        self.last_exp = np.exp(input_tensor - np.max(input_tensor, axis=1).reshape(len(input_tensor), 1))
        self.last_summed = self.last_exp.sum(axis=1).reshape(len(input_tensor), 1)
        return self.last_exp / self.last_summed

    def backward(self, error_tensor):
        amount = len(self.last_exp)
        classes = self.last_exp.shape[1]

        exp = np.repeat(self.last_exp.reshape(amount, 1, classes), classes, axis=1)
        transposed = exp.swapaxes(1, 2)
        summed = self.last_summed.reshape(amount, 1, 1)
        
        diagonal = np.diag(np.ones(classes))
        not_diagonal = 1.0*(diagonal == 0)

        differential = (diagonal * exp/summed) + (not_diagonal * 1/self.last_actual_sum) - (exp * transposed)/(summed**2)
        
        return (error_tensor.reshape(amount, 1, classes) @ differential).reshape(amount, classes)
