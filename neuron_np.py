import numpy as np

inputs = np.array(list(map(float, input().split())))
weights = np.array(list(map(float, input().split())))
bias = float(input())
prefinal = np.dot(inputs, weights)
output = prefinal + bias
print(f'Знач. нейрона: {output}; размер inputs {inputs.shape}, размер weights {weights.shape}')