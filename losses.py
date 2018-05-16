import numpy as np

class loss(object):
    def __init__(self, X):
        try:
            if len(X) == 2:
                self.y = X[0]
                self.X = X[1]
                self.N = np.shape(self.X)[0]

            else:
                raise ValueError('In case of a regression X must be a list [y, X]')

        except TypeError:
            raise TypeError('In case of a regression X must be a list [y, X]')

    def loss(self, i, Theta):
        pass

    def mean_loss(self, *kargs):
        return np.mean([self.loss(i, *kargs) for i in range(self.N)])

    def grad(self, i, Theta):
        pass

    def complete_grad(self, *kargs):
        grad = self.grad(0, *kargs)
        for i in range(1, self.N):
            grad += self.grad(i, *kargs)

        return grad / self.N

    def hess(self, i, *kargs):
        pass

    def complete_hess(self, *kargs, S=None):
        hess = self.hess(0, *kargs)
        for i in range(1, self.N):
            hess += self.hess(i, *kargs)

        if S is not None:
            return np.dot(hess / self.N, S)

        else:
            return hess / self.N


class reg_L2(loss):
    def __init__(self, X):
        super().__init__(X)

    def loss(self, i, Theta):
        y = self.y[i]
        x = self.X[i, :]
        return (y - np.dot(x, Theta))**2

    def grad(self, i, Theta):
        y = self.y[i]
        x = self.X[i, :]
        return -2 * x * (y - np.dot(x, Theta))

    def hess(self, i, *kargs):
        x = self.X[[i], :].reshape(np.shape(self.X)[1] ,1)
        return 2 * np.dot(x, np.transpose(x))


class SVM_loss(loss):
    def __init__(self, X):
        super().__init__(X)

    def loss(self, i, Theta):
        y = self.y[i]
        x = self.X[i, :]
        vker = np.vectorize(lambda _: self.kernel(x, _
                                                  ))
        return -2 * self.y[i] * Theta[i] +  Theta[i] * np.sum()
