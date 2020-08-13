import numpy as np
import math
from scipy import signal

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = tuple(stride_shape) if type(stride_shape) is not int else (stride_shape, stride_shape)
        self.num_kernels = num_kernels
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_kernels = None
        self._optimizer_biases = None

    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    def set_gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    def get_optimizer(self):
        return (self._optimizer_kernels, self._optimizer_biases)

    def set_optimizer(self, optimizer):
        self._optimizer_kernels = optimizer
        self._optimizer_biases = optimizer.deep_copy()

    optimizer = property(get_optimizer, set_optimizer)

    gradient_weights = property(get_gradient_weights, set_gradient_weights)
    gradient_bias = property(get_gradient_bias, set_gradient_bias)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.weights.size//self.weights.shape[0]
        fan_out = self.weights.size//self.weights.shape[1]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.last_input_shape = input_tensor.shape
        amount = len(input_tensor)
        channels = input_tensor.shape[1]
        
        paddedrest = tuple(inp + ker-1 for inp, ker in zip(input_tensor.shape[2:], self.weights.shape[2:]))
        self.last_input_padded = np.zeros((amount, channels, *paddedrest))
        if len(input_tensor.shape) == 3:
            pad = self.weights.shape[2]//2
            padend = self.weights.shape[2] - pad - 1
            padend = None if padend == 0 else -padend
            self.last_input_padded[:, :, pad:padend] = input_tensor
        else:
            pady = self.weights.shape[2]//2
            padyend = self.weights.shape[2] - pady - 1
            padyend = None if padyend == 0 else -padyend
            padx = self.weights.shape[3]//2
            padxend = self.weights.shape[3] - padx - 1
            padxend = None if padxend == 0 else -padxend
            self.last_input_padded[:, :, pady:padyend, padx:padxend] = input_tensor
        
        outshape = tuple(math.ceil(size/self.stride_shape[i]) for i, size in enumerate(input_tensor.shape[2:]))
        #print(outshape)
        output_tensor = np.empty((amount, self.num_kernels, *outshape))
        for i, sample in enumerate(input_tensor):
            for j, kernel in enumerate(self.weights):
                #print(sample.shape, kernel.shape)
                fullcorr = signal.correlate(sample, kernel, mode='same')[channels//2]
                if len(input_tensor.shape) == 3:
                    output_tensor[i, j] = fullcorr[::self.stride_shape[0]] + self.bias[j]
                else:
                    output_tensor[i, j] = fullcorr[::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[j]

        return output_tensor

    def backward(self, error_tensor):
        last_dims = (2,) if len(error_tensor.shape) == 3 else (2, 3)
        self._gradient_bias = error_tensor.sum(axis=(0, *last_dims))
        if self._optimizer_biases is not None:
            self.bias = self._optimizer_biases.calculate_update(self.bias, self._gradient_bias)
        
        upsampled_error = np.zeros((*error_tensor.shape[:2], *self.last_input_shape[2:]))

        if len(last_dims) == 1:
            upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor
        else:
            upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        self._gradient_weights = np.zeros_like(self.weights)
        #print(self.last_input_shape, self.stride_shape, error_tensor.shape, self.weights.shape)
        for i, sample in enumerate(self.last_input_padded):
            for k, channel in enumerate(sample):
                for j in range(len(self._gradient_weights)):
                    self._gradient_weights[j, k] += signal.correlate(channel, upsampled_error[i, j], mode='valid')

        if self._optimizer_kernels is not None:
            self.weights = self._optimizer_kernels.calculate_update(self.weights, self._gradient_weights)

        backkernels = np.swapaxes(self.weights, 0, 1)
        differential = np.empty(self.last_input_shape)
        #print(differential.shape, upsampled_error.shape, backkernels.shape)
        for i, error, in enumerate(upsampled_error):
            for j, kernel, in enumerate(backkernels):
                if len(last_dims) == 1:
                    differential[i, j] = signal.convolve(error, kernel, mode='same')[self.num_kernels//2]
                else:
                    differential[i, j] = signal.convolve(error, kernel[::-1], mode='same')[self.num_kernels//2]
        
        return differential