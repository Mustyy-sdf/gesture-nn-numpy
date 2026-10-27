import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        return self.output
    def backward(self, dvalues):
        # dvalues — реакция loss на каждый выход слоя, размер (примеры, нейроны)

        # Наклоны весов: вход × реакция, сложенные по всем примерам.
        # (входы, примеры) · (примеры, нейроны) → (входы, нейроны), как у self.weights
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Наклоны смещений: 1 × реакция, сложенные по примерам (axis=0).
        # keepdims=True оставляет размер (1, нейроны), как у self.biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Реакция для предыдущего слоя: вес × реакция, сложенные по нейронам.
        # (примеры, нейроны) · (нейроны, входы) → (примеры, входы), как у self.inputs
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
