import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size + 1)
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        self.last_input = input_tensor
        return (self.weights[:, :-1] @ input_tensor.transpose()).transpose() + self.weights[:, -1]

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def get_optimizer(self):
        return self._optimizer

    optimizer = property(get_optimizer, set_optimizer)

    def backward(self, error_tensor):
        input_size = self.last_input.shape[1]
        output_size = error_tensor.shape[1]
        amount = self.last_input.shape[0]
        gradient = np.empty_like(self.weights)
        gradient[:, -1] = error_tensor.sum(axis=0)
        differential = error_tensor.reshape(amount, output_size, 1) @ self.last_input.reshape(amount, 1, input_size)
        gradient[:, :-1] = differential.sum(axis=0)
        self.gradient_weights = gradient

        next_differential = error_tensor @ self.weights[:, :-1]
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return next_differential

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    gradient_weights = property(get_gradient_weights, set_gradient_weights)
