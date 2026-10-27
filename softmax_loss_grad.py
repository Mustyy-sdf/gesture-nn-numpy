import numpy as np
from nn.layers import Layer_Dense
from nn.activations import Activation_ReLU
from nn.activations import Activation_Softmax
from nn.losses import Loss_CategoricalCrossentropy
from spiral import spiral_data
p = np.array([[0.7,  0.1, 0.2],
              [0.1,  0.5, 0.4],
              [0.02, 0.9, 0.08]])
y = np.array([0, 1, 1])

n = len(p)               # число примеров (строк)
grad = p.copy()          # копия вероятностей, чтобы не испортить p
grad[range(n), y] -= 1   # в каждой строке у правильного жеста вычитаем 1
grad = grad / n          # делим на число примеров: loss — это среднее
print(grad)
