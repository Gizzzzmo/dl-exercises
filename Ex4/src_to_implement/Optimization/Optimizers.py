import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor

    def regularizer_loss(self, *weights):
        return 0 if self.regularizer is None or weights[0] is None else self.regularizer.norm(weights[0])

    def deep_copy(self):
        copy = type(self)()
        copy.add_regularizer(self.regularizer)

class OptimizerWrapper(Optimizer):
    def __init__(self, *optimizers):
        super().__init__()
        self.optimizers = optimizers

    def calculate_update(self, *weight_gradient_pairs):
        return [opt.calculate_update(w_t, g_t) for opt, (w_t, g_t) in zip(self.optimizers, weight_gradient_pairs)]

    def regularizer_loss(self, *weights):
        return sum(opt.regularizer_loss(w) for opt, w in zip(self.optimizers, weights))

    def add_regularizer(self, *regularizer):
        if len(regularizer) == 1:
            for opt in self.optimizers:
                opt.add_regularizer(regularizer[0])
        else:
            for opt, reg in zip(self.optimizers, regularizer):
                opt.add_regularizer(reg)


class Sgd(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        v = -self.lr * gradient_tensor
        
        return weight_tensor + v

    def deep_copy(self):
        sgd = Sgd(self.lr)
        sgd.add_regularizer(self.regularizer)
        return sgd

class SgdWithMomentum(Optimizer):
    def __init__(self, lr, mr):
        super().__init__()
        self.lr = lr
        self.mr = mr
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        #if self.regularizer is not None:
        #    gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        self.v = self.mr * self.v - self.lr * gradient_tensor
        
        if self.regularizer is None:
            return weight_tensor + self.v
        else:
            return weight_tensor + self.v - self.lr * self.regularizer.calculate_gradient(weight_tensor)

    def deep_copy(self):
        sdgWM = SgdWithMomentum(self.lr, self.mr)
        sdgWM.v = self.v
        sdgWM.add_regularizer(self.regularizer)
        return sdgWM

class Adam(Optimizer):
    def __init__(self, lr, mu, rho):
        super().__init__()
        self.lr = lr
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        g = gradient_tensor

        #if self.regularizer is not None:
        #    g += self.regularizer.calculate_gradient(weight_tensor)

        self.v = self.mu*self.v + (1-self.mu)*g
        self.r = self.rho*self.r + (1-self.rho) * g * g

        v_hat = self.v/(1 - self.mu**self.k)
        r_hat = self.r/(1 - self.rho**self.k)
        self.k += 1

        update = (v_hat + np.finfo(float).eps)/(np.sqrt(r_hat) + np.finfo(float).eps)

        if self.regularizer is None:
            return weight_tensor - self.lr * update
        else:
            return weight_tensor - self.lr * (self.regularizer.calculate_gradient(weight_tensor) + update)

    def deep_copy(self):
        adam = Adam(self.lr, self.mu, self.rho)
        adam.v = self.v
        adam.r = self.r
        adam.k = self.k
        adam.add_regularizer(self.regularizer)
        return adam