from functools import reduce

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        inputs, labels = self.data_layer.forward()
        self.last_labels = labels
        lastoutput = reduce(lambda result, layer: layer.forward(result), [inputs] + self.layers)
        return self.loss_layer.forward(lastoutput, labels)

    def backward(self):
        differential = self.loss_layer.backward(self.last_labels)
        lastdifferential = reduce(lambda result, layer: layer.backward(result), [differential] + self.layers[::-1])
        return lastdifferential

    def append_trainable_layer(self, layer):
        layer.optimizer = self.optimizer.deep_copy()
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            self.loss.append(self.forward())

    def test(self, input_tensor):
        return reduce(lambda result, layer: layer.forward(result), [input_tensor] + self.layers)
