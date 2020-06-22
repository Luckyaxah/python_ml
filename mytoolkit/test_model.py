import numpy as np

from model import LinearRegressionModel
def test_1dimension_x():
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

def test_2dimension_x():

    x = np.array([
        [152,50],
        [183,20],
        [171,20],
        [165,30],
        [158,30],
        [161,50],
        [149,60],
        [158,50],
        [170,40],
        [153,55],
        [164,40],
        [190,40],
        [185,20]
    ])

    y = np.array([
        [120],
        [141],
        [124],
        [126],
        [117],
        [125],
        [123],
        [125],
        [132],
        [123],
        [132],
        [155],
        [147]
    ])

    model = LinearRegressionModel(x, y)
    assert (model.observed_x == x).all()
    assert (model.observed_y == y).all()
    
    
    alpha = 3.3*1e-5
    model.train(method='bgd',alpha=alpha, max_iterations1=300000, initial_params=np.array([-62.96,1.068,0.4]))
    print(model.error)
    print(model.params)

    # 最小二乘的结果：-62.963, 1.068, 0.4


    # import matplotlib.pyplot as plt
    # plt.plot(model.errors)
    # plt.show()




test_2dimension_x()