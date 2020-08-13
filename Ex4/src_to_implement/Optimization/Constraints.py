import numpy as np

class Norm_Regularizer:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

class L2_Regularizer(Norm_Regularizer):
    def __init__(self, alpha):
        super().__init__(alpha)

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.sqrt(np.sum(np.square(weights)))

class L1_Regularizer(Norm_Regularizer):
    def __init__(self, alpha):
        super().__init__(alpha)

    def calculate_gradient(self, weights):
        return self.alpha* np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))