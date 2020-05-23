import numpy as np

class Sgd:
    def __init__(self, lr):
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.lr * gradient_tensor

    def deep_copy(self):
        return Sgd(self.lr)

class SgdWithMomentum:
    def __init__(self, lr, mr):
        self.lr = lr
        self.mr = mr
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mr * self.v - self.lr * gradient_tensor
        
        return weight_tensor + self.v

    def deep_copy(self):
        sdgWM = SgdWithMomentum(self.lr, self.mr)
        sdgWM.v = self.v
        return sdgWM

class Adam:
    def __init__(self, lr, mu, rho):
        self.lr = lr
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        g = gradient_tensor
        self.v = self.mu*self.v + (1-self.mu)*g
        self.r = self.rho*self.r + (1-self.rho) * g * g

        v_hat = self.v/(1 - self.mu**self.k)
        r_hat = self.r/(1 - self.rho**self.k)
        self.k += 1

        return weight_tensor - self.lr * (v_hat + np.finfo(float).eps)/(np.sqrt(r_hat) + np.finfo(float).eps)

    def deep_copy(self):
        adam = Adam(self.lr, self.mu, self.rho)
        adam.v = self.v
        adam.r = self.r
        adam.k = self.k
        return adam