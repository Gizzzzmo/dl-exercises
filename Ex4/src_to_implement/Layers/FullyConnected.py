import numpy as np
from .Base import Layer

class FullyConnected(Layer):
    def __init__(self, input_size, output_size, with_bias=True, phase='train'):
        super().__init__(phase)
        self._with_bias = with_bias
        self.weights = np.random.rand(output_size, input_size + (1 if with_bias else 0))
        self._gradient_weights = None

    def forward(self, input_tensor):
        self.last_input = input_tensor
        return (self.weights[:, :input_tensor.shape[1]] @ input_tensor.transpose()).transpose() + (self.weights[:, -1] if self._with_bias else 0)

    def backward(self, error_tensor):
        input_size = self.last_input.shape[1]
        output_size = error_tensor.shape[1]
        amount = self.last_input.shape[0]
        gradient = np.empty_like(self.weights)
        if self._with_bias:
            gradient[:, -1] = error_tensor.sum(axis=0)
        differential = error_tensor.reshape(amount, output_size, 1) @ self.last_input.reshape(amount, 1, input_size)
        gradient[:, :input_size] = differential.sum(axis=0)
        
        if self.accumulate_gradients and self.gradient_weights is not None:
            self._gradient_weights += gradient
        else:
            self._gradient_weights = gradient

        next_differential = error_tensor @ self.weights[:, :input_size]
        if self.optimizer is not None and not self.accumulate_gradients:
            self.update_weights()
        return next_differential

    def update_weights(self):
        self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        self._gradient_weights[:] = 0

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.weights.shape[1] - (1 if self._with_bias else 0)
        fan_out = self.weights.shape[0]
        if self._with_bias:
            self.weights[:, :-1] = weights_initializer.initialize((fan_out, fan_in), fan_in, fan_out)
            self.weights[:, -1] = bias_initializer.initialize((fan_out,), fan_in, fan_out)
        else:
            self.weights = weights_initializer.initialize((fan_out, fan_in), fan_in, fan_out)