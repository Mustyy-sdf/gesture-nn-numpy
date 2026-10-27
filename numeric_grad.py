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
old_loss = 9999999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
# loss до
out1 = activation1.forward(dense1.forward(X))
out2 = activation2.forward(dense2.forward(out1))
loss_before = loss_function.calculate(out2, y)

# чуть сдвинуть один вес
dense2.weights[0, 0] += 0.0001

# loss после
out1 = activation1.forward(dense1.forward(X))
out2 = activation2.forward(dense2.forward(out1))
loss_after = loss_function.calculate(out2, y)

# наклон и возврат веса
naklon = (loss_after - loss_before) / 0.0001
dense2.weights[0, 0] -= 0.0001
print(naklon)
    