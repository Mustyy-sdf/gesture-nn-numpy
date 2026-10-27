inputs = list(map(float, input().split()))
weights = list(map(float, input().split()))
if len(inputs) == len(weights):
  bias = float(input())
  prefinal = list()
  for x, w in zip(inputs, weights):
      prefinal.append(x * w)
  output = sum(prefinal) + bias
  print(output)
else:
  print('len(inputs) != len(weights)')