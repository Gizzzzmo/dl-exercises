
class Layer:
    def __init__(self, phase='train'):
        self._phase = phase
        self._optimizer = None
        self._accumulate_gradients = False
        self.weights = None
    @property
    def testing_phase(self):
        return self._phase == Phase.test

    @testing_phase.setter
    def testing_phase(self, test):
        self._phase = Phase.test if test else Phase.train

    def get_phase(self):
        return self._phase

    def set_phase(self, phase):
        self._phase = phase

    def get_optimizer(self):
        return self._optimizer
    
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def get_accumulate(self):
        return self._accumulate_gradients

    def set_accumulate(self, acc):
        self._accumulate_gradients = acc

    def update_weights(self):
        pass

    phase = property(lambda self: self.get_phase(), lambda self, phase: self.set_phase(phase))
    optimizer = property(lambda self: self.get_optimizer(), lambda self, optimizer: self.set_optimizer(optimizer))
    accumulate_gradients = property(lambda self: self.get_accumulate(), lambda self, acc: self.set_accumulate(acc))

    def initialize(self, weights_initializer, bias_initializer):
        pass

    def regularizer_loss(self):
        return self.optimizer.regularizer_loss(self.weights) if self.optimizer is not None else 0

class Phase:
    train = 'train'
    test = 'test'