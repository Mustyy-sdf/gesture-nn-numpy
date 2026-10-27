import numpy as np
from nn.layers import Layer_Dense
from nn.activations import Activation_ReLU
from nn.activations import Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
dense1 = Layer_Dense(42, 64)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
out1 = activation1.forward(dense1.forward(np.random.randn(500, 42)))
dense2 = Layer_Dense(64, 6)
out2 = activation2.forward(dense2.forward(out1))
y = np.random.randint(0, 6, size=500)
loss_function = Loss_CategoricalCrossentropy()
out3 = loss_function.calculate(out2, y)
print(out3)