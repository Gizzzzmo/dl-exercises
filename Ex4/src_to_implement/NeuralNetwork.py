from functools import reduce
import pickle as pkl
from Layers.Base import Phase

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

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
        layer.initialize(self.weights_initializer, self.bias_initializer)
        layer.optimizer = self.optimizer.deep_copy()
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = Phase.train
        for i in range(iterations):
            self.loss.append(self.forward())
            self.loss[-1] = reduce(lambda reg_loss, layer: reg_loss + layer.regularizer_loss(), [self.loss[-1]] + self.layers)
            print('iteration', i, 'loss:', self.loss[-1])
            self.backward()

    def test(self, input_tensor):
        self.phase = Phase.test
        output = reduce(lambda result, layer: layer.forward(result), [input_tensor] + self.layers)
        return output

    def set_phase(self, phase):
        for layer in self.layers:
            layer.phase = phase

    def get_phase(self):
        if not self.layers:
            return None
        else:
            return self.layers[0].phase
    
    phase = property(get_phase, set_phase)

    def __getstate__(self):
        copy = self.__dict__.copy()
        del copy['data_layer']
        return copy

    def __setstate__(self, state):
        self.__dict__ = state
        self.data_layer = None

def save(filename, net):
    with open(filename, 'wb') as f:
        pkl.dump(net, f)

def load(filename, data_layer):
    with open(filename, 'rb') as f:
        net = pkl.load(f)
    net.data_layer = data_layer
    return net