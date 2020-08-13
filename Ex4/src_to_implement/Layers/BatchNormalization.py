from .Base import Layer, Phase
from Layers.Helpers import compute_bn_gradients
from Optimization.Optimizers import OptimizerWrapper
import numpy as np

class BatchNormalization(Layer):
    alpha = 0.8
    def __init__(self, channels):
        super().__init__(phase='train')
        self.channels = channels
        self._gradient_weights = None
        self._gradient_bias = None
        self._weights_optimizer = None
        self._bias_optimizer = None
        self.last_samples = None
        self.initialize(None, None)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((self.channels,))
        self.bias = np.zeros_like(self.weights)
        self.moving_avg_weights = None
        self.moving_avg_bias = None

    def format_flatten(self, tensor, samples, final_shape):
        tensor = np.swapaxes(tensor.reshape(samples, self.channels, -1), 1, 2)
        return tensor.reshape(tensor.shape[0]*tensor.shape[1], -1)

    def format_rollup(self, tensor, samples, final_shape):
        tensor = np.swapaxes(tensor.reshape(samples, int(np.prod(final_shape)), -1), 1, 2)
        return tensor.reshape(samples, self.channels, *final_shape)

    def reformat(self, tensor):
        if(len(tensor.shape) == 2):
            if self.last_samples is None:
                self.last_final_shape = self.input_shape[2:]
                self.last_samples = self.input_shape[0]
            return self.format_rollup(tensor, self.last_samples, self.last_final_shape)
        else:
            self.last_samples = len(tensor)
            self.last_final_shape = tensor.shape[2:]
            return self.format_flatten(tensor, len(tensor), tensor.shape[2:])

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def get_optimizer(self):
        return OptimizerWrapper(self._weights_optimizer, self._bias_optimizer)

    def set_optimizer(self, optimizer):
        self._weights_optimizer = optimizer
        self._bias_optimizer = optimizer.deep_copy()

    def regularizer_loss(self):
        return self.optimizer.regularizer_loss(self.weights, self.bias)

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        final_shape = input_tensor.shape[2:]
        samples = input_tensor.shape[0]
        input_tensor = self.format_flatten(input_tensor, samples, final_shape)

        self.last_input_tensor = input_tensor

        if self.phase == Phase.train:
            mean = np.mean(input_tensor, axis=0)
            std = np.std(input_tensor, axis=0)
            self.last_mean = mean
            self.last_std = std

            if self.moving_avg_weights is None:
                self.moving_avg_weights = std
                self.moving_avg_bias = mean
            else:
                self.moving_avg_weights *= self.alpha
                self.moving_avg_bias *= self.alpha
                self.moving_avg_weights += (1-self.alpha) * std
                self.moving_avg_bias += (1-self.alpha) * mean
        else:
            mean = self.moving_avg_bias
            std = self.moving_avg_weights

        output_tensor = (input_tensor - mean)/np.sqrt(std**2 + 1e-10)

        self.last_input_tilde = output_tensor
        output_tensor = self.weights * output_tensor + self.bias

        return self.format_rollup(output_tensor, samples, final_shape)

    def backward(self, error_tensor):
        final_shape = error_tensor.shape[2:]
        samples = error_tensor.shape[0]

        error_tensor = self.format_flatten(error_tensor, samples, final_shape)

        if self.accumulate_gradients and self._gradient_weights is not None:
            self._gradient_weights += np.sum(error_tensor * self.last_input_tilde, axis=0)
            self._gradient_bias += np.sum(error_tensor, axis=0)
        else:
            self._gradient_weights = np.sum(error_tensor * self.last_input_tilde, axis=0)
            self._gradient_bias = np.sum(error_tensor, axis=0)

        if self._weights_optimizer is not None and not self.accumulate_gradients:
            self.update_weights()

        differential = compute_bn_gradients(error_tensor, self.last_input_tensor, self.weights, self.last_mean, self.last_std**2)

        return self.format_rollup(differential, samples, final_shape)

    def update_weights(self):
        self.weights = self._weights_optimizer.calculate_update(self.weights, self._gradient_weights)
        self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        self.gradient_weights[:] = 0
        self.gradient_bias[:] = 0
