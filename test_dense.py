import numpy as np
from nn.layers import Layer_Dense

# Те же данные, что в layer.py: 3 примера по 4 числа
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

# Веса из layer.py записаны как (нейроны, входы) = (3, 4)
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([[2.0, 3.0, 0.5]])  # (1, 3): по одному смещению на нейрон

# Ответ, который layer.py уже посчитал вручную
expected = np.array([[4.8, 1.21, 2.385],
                     [8.9, -1.81, 0.2],
                     [1.41, 1.051, 0.026]])

dense = Layer_Dense(4, 3)
dense.weights = weights.T  # класс хранит (входы, нейроны) = (4, 3), поэтому .T
dense.biases = biases
dense.forward(inputs)

print('Ваш Layer_Dense')
print('  размер правильный:', dense.output.shape == expected.shape)
print('  числа совпали:    ', np.allclose(dense.output, expected))

# Демонстрация: что было бы, если бы в forward забыли прибавить смещения
broken = np.dot(inputs, dense.weights)

print('Слой, в котором забыли смещения')
print('  размер правильный:', broken.shape == expected.shape)
print('  числа совпали:    ', np.allclose(broken, expected))

# Проверка backward: реакция придумана для теста, ответы посчитаны заранее
dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])
dense.backward(dvalues)

expected_dweights = np.array([[0.5, 0.5, 0.5],
                              [20.1, 20.1, 20.1],
                              [10.9, 10.9, 10.9],
                              [4.1, 4.1, 4.1]])
expected_dbiases = np.array([[6.0, 6.0, 6.0]])
expected_dinputs = np.array([[0.44, -0.38, -0.07, 1.37],
                             [0.88, -0.76, -0.14, 2.74],
                             [1.32, -1.14, -0.21, 4.11]])

print('Layer_Dense.backward')
print('  dweights совпали:', np.allclose(dense.dweights, expected_dweights))
print('  dbiases совпали: ', np.allclose(dense.dbiases, expected_dbiases))
print('  dinputs совпали: ', np.allclose(dense.dinputs, expected_dinputs))
