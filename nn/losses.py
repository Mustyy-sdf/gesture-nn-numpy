import numpy as np

class Loss_CategoricalCrossentropy:
  def forward(self, y_pred, y_true):
    self.y_pred = y_pred
    self.y_true = y_true
    result = -np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)[range(y_pred.shape[0]), y_true])
    return result
  def calculate(self, y_pred, y_true):
    return np.mean(self.forward(y_pred, y_true))
  def accuracy(self, y_pred, y_true):
    self.y_pred = y_pred
    self.y_true = y_true
    return np.mean(np.argmax(y_pred, axis=1) == y_true)