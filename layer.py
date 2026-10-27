import numpy as np
# def output(inputs, weights, biases):
#   prefinal = []
#   for w in weights:
#     prefinal.append(sum(w*inputs))
#   return np.array(prefinal) + biases

def relu(x):
    return np.maximum(x, 0)

inputs = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
    ])

weights = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
    ])

biases = np.array([2.0, 3.0, 0.5])

print(relu(np.dot(inputs, weights.T) + biases), inputs.shape, inputs.T.shape, weights.shape, biases.shape)
#print(output(inputs, weights, biases))
