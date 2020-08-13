from .Base import Layer

class ReLU(Layer):    
    def forward(self, input_tensor):
        self.last_derivative = input_tensor > 0
        return input_tensor * self.last_derivative

    def backward(self, error_tensor):
        return error_tensor * self.last_derivative
