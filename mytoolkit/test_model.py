import numpy as np

from model import LinearRegressionModel

x = np.array([
    [1],
    [0.5],
    [1.5],
    [2]
])

y = np.array([
    [-0.9],
    [-0.11],
    [0.28],
    [0.5]
])

model = LinearRegressionModel(x, y)
model.params = np.array([1,1])[:, np.newaxis]
assert (model.observed_x == x).all()
assert (model.observed_y == y).all()

model.train(method='bgd')
print(model.grad)


