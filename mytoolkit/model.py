import numpy as np

class LinearRegressionModel:
    def __init__(self, features, target):
        m, n = features.shape
        target = LinearRegressionModel.to_col_vec(target)
        m1, n1 = target.shape
        if m1 != m or n1 != 1:
            raise ValueError
        self.sample_num = m
        self.features_num = n
        self.params_num = self.features_num + 1
        self.observed_x = features
        self.observed_y = target
        self.params = []
        self.error = None
        self.errors = []
    
    def train(self, method='bgd', **kwargs):
        x = np.concatenate( (np.ones((self.sample_num,1)), self.observed_x), axis = 1)
        if method == 'bgd':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = 0.01
            if 'max_iterations1' in kwargs:
                max_iterations1 = kwargs['max_iterations1']
            else:
                max_iterations1 = 10000
            if 'initial_params' in kwargs:
                self.params = LinearRegressionModel.to_col_vec(kwargs['initial_params'])
            else:
                self.params = np.array([np.random.rand() for i in range(self.params_num)])[:, np.newaxis]
            count = 0
            converged_flag = False
            epsilon = 1e-5
            self.errors = []
            while True:
                count += 1
                # 梯度的负方向
                self.error = np.sum((np.dot(x, self.params) - self.observed_y) ** 2)/ self.sample_num
                self.errors.append(self.error)
                if(self.error < epsilon):
                    converged_flag = True
                if converged_flag or max_iterations1< count:
                    break
                self.grad = [0 for j in range(self.params_num)]
                for j in range(self.params_num):
                    y_diff = np.dot(x, self.params) - self.observed_y
                    x_j = x[:,j][:,np.newaxis]
                    self.grad[j] = 2 * np.sum(np.multiply(y_diff, x_j))/ self.sample_num
                self.grad = np.array(self.grad)[:, np.newaxis]
                self.params = self.params - alpha * self.grad

        elif method == 'sbgd':
            pass

    def predict(self, x):
        x = np.concatenate((np.ones((self.sample_num, 1)), x), axis=1)
        return np.dot(x, self.params)

    @staticmethod
    def to_col_vec(arr):
        if type(arr) != np.ndarray:
            raise TypeError
        elif len(arr.shape) == 1:
            arr = arr[:, np.newaxis]
        elif len(arr.shape) == 2 and arr.shape[1] != 1:
            if arr.shape[0] == 1:
                arr = arr.T
            else:
                raise ValueError
        elif len(arr.shape) == 2 and arr.shape[1] == 1:
            pass
        else:
            raise ValueError
        return arr
    