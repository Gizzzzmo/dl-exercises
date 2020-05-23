import numpy as np
import math

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = tuple(stride_shape) if type(stride_shape) is not int else (stride_shape, stride_shape)
        self.pooling_shape = tuple(pooling_shape) if type(pooling_shape) is not int else (pooling_shape, pooling_shape)

    def forward(self, input_tensor):
        last_dims = tuple( math.ceil((inshape - pool + 1)/stride) for stride, pool, inshape in zip(self.stride_shape, self.pooling_shape, input_tensor.shape[2:]))
        output_tensor = np.empty((*input_tensor.shape[0:2], *last_dims))
        self.last_index_tensor = np.empty_like(output_tensor, dtype=int)
        self.last_shape = input_tensor.shape
        
        for i in range(output_tensor.shape[2]):
            for j in range(output_tensor.shape[3]):
                ystart = i*self.stride_shape[0]
                yend = ystart + self.pooling_shape[0]
                xstart = j*self.stride_shape[1]
                xend = xstart + self.pooling_shape[1]
                pool = input_tensor[:, :, ystart:yend, xstart:xend].reshape(len(input_tensor), input_tensor.shape[1], -1)
                output_tensor[:, :, i, j] = np.max(pool, axis=2)
                self.last_index_tensor[:, :, i, j] = np.argmax(pool, axis=-1)

        return output_tensor

    def backward(self, error_tensor):
        differential = np.zeros(self.last_shape)
        for k, sample_error in enumerate(error_tensor):
            for l, pool_error in enumerate(sample_error):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        ystart = i*self.stride_shape[0]
                        xstart = j*self.stride_shape[1]
                        index = self.last_index_tensor[k, l, i, j]
                        yindex = ystart + index//self.stride_shape[0]
                        xindex = xstart + index%self.stride_shape[0]
                        differential[k, l, yindex, xindex] +=pool_error[i, j]
                        
        return differential
