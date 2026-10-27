import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(self.inputs, 0)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        mask = self.inputs <= 0
        self.dinputs[mask] = 0
        return self.dinputs

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        exp_sum = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / exp_sum
        return self.output