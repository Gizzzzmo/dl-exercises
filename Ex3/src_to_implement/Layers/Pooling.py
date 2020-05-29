import numpy as np
import math

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = tuple(stride_shape) if type(stride_shape) is not int else (stride_shape, stride_shape)
        self.pooling_shape = tuple(pooling_shape) if type(pooling_shape) is not int else (pooling_shape, pooling_shape)

    def forward(self, input_tensor):
        last_dims = tuple( math.ceil((inshape - pool + 1)/stride) for stride, pool, inshape in zip(self.stride_shape, self.pooling_shape, input_tensor.shape[2:]))
        output_tensor = np.empty((*input_tensor.shape[0:2], *last_dims))
        self.last_index_y = np.empty_like(output_tensor, dtype=int)
        self.last_index_x = np.empty_like(output_tensor, dtype=int)
        self.last_shape = input_tensor.shape
        
        for i in range(output_tensor.shape[2]):
            for j in range(output_tensor.shape[3]):
                ystart = i*self.stride_shape[0]
                yend = ystart + self.pooling_shape[0]
                xstart = j*self.stride_shape[1]
                xend = xstart + self.pooling_shape[1]
                pool = input_tensor[:, :, ystart:yend, xstart:xend].reshape(len(input_tensor), input_tensor.shape[1], -1)
                output_tensor[:, :, i, j] = np.max(pool, axis=-1)
                
                index_y, index_x = np.unravel_index(np.argmax(pool, axis=-1), self.pooling_shape)
                self.last_index_y[:, :, i, j] = index_y + ystart
                self.last_index_x[:, :, i, j] = index_x + xstart

        return output_tensor

    def backward(self, error_tensor):

        differential = np.zeros(self.last_shape)
        ind_samples = np.tile(np.arange(error_tensor.shape[0]), (error_tensor.shape[1], 1)).transpose()
        ind_channels = np.tile(np.arange(error_tensor.shape[1]), (error_tensor.shape[0], 1))
        
        for i in range(error_tensor.shape[2]):
            for j in range(error_tensor.shape[3]):
                differential[ind_samples, ind_channels, self.last_index_y[:, :, i, j], self.last_index_x[:, :, i, j]] += error_tensor[:, :, i, j]
                                
        return differential
