import numpy as np
import pickle as pkl
from NeuralNetwork import NeuralNetwork
from Optimization.Optimizers import Adam
from Optimization.Loss import CrossEntropyLoss
from Layers import *
from Layers.Initializers import He

def build():
    net = NeuralNetwork(Adam(5e-4, 0.9, 0.999), He(), He())
    net.append_trainable_layer(Conv.Conv(1, (1, 5, 5), 20))
    net.append_trainable_layer(Pooling.Pooling(2, 2))
    net.append_trainable_layer(ReLU.ReLU())
    net.append_trainable_layer(Conv.Conv(1, (20, 5, 5), 50))
    net.append_trainable_layer(Pooling.Pooling(2, 2))
    net.append_trainable_layer(ReLU.ReLU())
    net.append_trainable_layer(Flatten.Flatten())
    net.append_trainable_layer(FullyConnected.FullyConnected(50*7*7, 500))
    net.append_trainable_layer(ReLU.ReLU())
    net.append_trainable_layer(FullyConnected.FullyConnected(500, 10))
    net.append_trainable_layer(SoftMax.SoftMax())
    net.loss_layer = CrossEntropyLoss()
    return net