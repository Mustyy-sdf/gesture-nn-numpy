import numpy as np
from nn.activations import Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy

z = np.array([[2.0, 1.0, 0.1]])   # входы Softmax
y = np.array([0])                 # правильный жест
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

# loss до
out2 = activation2.forward(z)
loss_before = loss_function.calculate(out2, y)

# чуть сдвинуть один вход Softmax
z[0, 0] += 0.0001

# loss после
out2 = activation2.forward(z)
loss_after = loss_function.calculate(out2, y)

# наклон и возврат
naklon = (loss_after - loss_before) / 0.0001
z[0, 0] -= 0.0001
print(naklon)