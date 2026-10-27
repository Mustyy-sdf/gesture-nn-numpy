import numpy as np
from nn.layers import Layer_Dense
from nn.activations import Activation_ReLU
from nn.activations import Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy
from spiral import spiral_data
np.random.seed(0)
X, y = spiral_data(100, 3)
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)
best_loss = 9999999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
for i in range(10**4):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    out1 = activation1.forward(dense1.forward(X))
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)
    out2 = activation2.forward(dense2.forward(out1))
    out3 = loss_function.calculate(out2, y)
    accurancy = loss_function.accuracy(out2, y)
    if out3 < best_loss:
        best_loss = out3
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        print(f'#{i}, best loss: {best_loss}, accuracy: {accurancy}')
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        
    