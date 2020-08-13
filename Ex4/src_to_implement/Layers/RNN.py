from .Base import Layer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid
from Optimization.Optimizers import OptimizerWrapper
import numpy as np

class RNN(Layer):
    def __init__(self, input_size, hidden_size, output_size, phase='train'):
        super().__init__(phase=phase)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False
        self._gradient_bias = None

        self.whh = FullyConnected(hidden_size, hidden_size, False, phase)
        self.wxh = FullyConnected(input_size, hidden_size, False, phase)
        self.bias = np.zeros(hidden_size)
        self.tanh = TanH(phase)
        self.why = FullyConnected(hidden_size, output_size, phase=phase)
        self.sigmoid = Sigmoid(phase)

        self.whh.accumulate_gradients = True

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize(self.bias.shape, self.input_size + self.hidden_size, self.hidden_size)
        self.whh.initialize(weights_initializer, bias_initializer)
        self.wxh.initialize(weights_initializer, bias_initializer)
        self.why.initialize(weights_initializer, bias_initializer)

    def set_accumulate(self, acc):
        super().set_accumulate(acc)
        self.why.accumulate_gradients = acc
        self.wxh.accumulate_gradients = acc

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def gradient_weights(self):
        return np.block([self.whh.gradient_weights, self.wxh.gradient_weights, self._gradient_bias.reshape(self.hidden_size, 1)])

    @property
    def weights(self):
        return np.block([self.whh.weights, self.wxh.weights, self.bias.reshape(self.hidden_size, 1)])

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self.whh.weights = None
            self.wxh.weights = None
            self.bias = None
        else:
            self.whh.weights = weights[:, :self.hidden_size]
            self.wxh.weights = weights[:, self.hidden_size:self.hidden_size + self.input_size]
            self.bias = weights[:, self.hidden_size + self.input_size]

    def set_phase(self, phase):
        super().set_phase(phase)
        self.wxh.phase = phase
        self.whh.phase = phase
        self.tanh.phase = phase
        self.why.phase = phase
        self.sigmoid.phase = phase

    def get_optimizer(self):
        return OptimizerWrapper(
                self.wxh.optimizer,
                self.whh.optimizer,
                self.why.optimizer,
                self._optimizer
            )

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self.wxh.optimizer = optimizer.deep_copy()
        self.whh.optimizer = optimizer.deep_copy()
        self.why.optimizer = optimizer.deep_copy()

    def regularizer_loss(self):
        wxh_loss = self.wxh.regularizer_loss()
        whh_loss = self.whh.regularizer_loss()
        why_loss = self.why.regularizer_loss()
        bias_loss = self._optimizer.regularizer_loss(self.bias)

        return whh_loss + wxh_loss + why_loss + bias_loss

    def forward(self, input_tensor):
        if not self.memorize:
            self.hidden_state = np.zeros_like(self.hidden_state)

        hidden_states = np.concatenate([self.hidden_state.reshape(1, -1), self.wxh.forward(input_tensor)])
        self.last_tanh_inputs = np.empty((len(input_tensor), self.hidden_size))
        for i in range(len(input_tensor)):
            hidden_states[i+1] = self.tanh.forward(self.whh.forward(hidden_states[i:i+1])[0] + hidden_states[i+1] + self.bias)

        self.hidden_state = hidden_states[-1]
        self.last_hidden_states = hidden_states
        self.last_wxh_inputs = input_tensor
        return self.sigmoid.forward(self.why.forward(hidden_states[1:]))

    def backward(self, error_tensor):
        why_error = self.why.backward(self.sigmoid.backward(error_tensor))

        tanh_error = np.zeros((len(error_tensor) + 1, self.hidden_size))

        if not self.accumulate_gradients:
            self.whh._gradient_weights = None

        for i in reversed(range(len(why_error))):
            self.tanh.last_tanh = self.last_hidden_states[i+1]
            tanh_error[i+1] = self.tanh.backward(tanh_error[i+1] + why_error[i])

            self.whh.last_input = self.last_hidden_states[i:i+1]
            tanh_error[i:i+1] = self.whh.backward(tanh_error[i+1:i+2])

        if self.accumulate_gradients and self._gradient_bias is not None:
            self._gradient_bias += tanh_error[1:].sum(axis=0)
        else:
            self._gradient_bias = tanh_error[1:].sum(axis=0)

        differential = self.wxh.backward(tanh_error[1:])

        if self._optimizer is not None and not self.accumulate_gradients:
            self.update_weights()
        
        return differential

    def update_weights(self):
        self.whh.update_weights()
        self.wxh.update_weights()
        self.why.update_weights()
        self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
        self._gradient_bias[:] = 0