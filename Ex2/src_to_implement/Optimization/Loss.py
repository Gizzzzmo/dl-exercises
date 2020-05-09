import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        input_tensor = input_tensor + (input_tensor == 0) * 13121e-145
        self.last_input = input_tensor
        return -(np.log(input_tensor) * label_tensor).sum()/len(input_tensor)

    def backward(self, label_tensor):
        differential = -(label_tensor/self.last_input)/len(label_tensor)
        return differential