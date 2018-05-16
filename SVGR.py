import numpy as np
import losses
import utils
import time
from memory_profiler import profile

class genral_model(object):
    '''
    General version of the model to be used for all models
    '''
    def __init__(self):
        self.theta_memory = []
        self.loss_memory = []
        self.time_memory = []
        self.d = None
        self.N = None
        self.K = None
        self.T = None
        self.Theta = None
        self.Loss = None

    #@profile
    def fit(self, X, K, T, learning_rate, loss='reg_L2', Theta_0=[-1, 1], decay=None, rate=None, save_at_every_step=False):
        '''
        :param X: Data (can contain y and X) (np.array)
        :param K: (int)
        :param T: (int)
        :param loss: (str)
        :param Theta_0: (np.array) If init is specified
                        ([low, high]) If not specified
        :param learning_rate: (str) for decaying learning rate
                              (int otherwise)
        :param rate: (float) rate of decay of the learning rate
        '''

        self.N = np.shape(X[1])[0]
        if loss == 'reg_L2':
            self.d = np.shape(X[1])[1]

        elif loss == 'SVM':
            self.d = self.N

        self.K = K
        self.T = T

        # Initialising the loss function, which will determine the kind of fit
        Loss = getattr(losses, loss)
        self.Loss = Loss(X)

        # Initialising Theta
        if isinstance(Theta_0, np.ndarray):
            if np.shape(Theta_0) == (self.d,):
                pass

            else:
                raise ValueError('If initialisation is specified, Theta_0 must either be an array of dimension (d,)')

        else:
            try:
                if len(Theta_0) == 2:
                    Theta_0 = np.random.uniform(low=Theta_0[0], high=Theta_0[1], size=self.d)

                else:
                    raise ValueError('For random initialization Theta must be of type [high, low]')

            except TypeError:
                raise TypeError('Theta_0 must either be an array of dimension (d,) or'
                                 'a list [high, low] for the uniform initialisation')


        # Initialising the learning rate
        assert isinstance(learning_rate, (int, float)), 'The learning_rate must be a number (float, int)'
        if rate is not None or decay is not None:
            assert rate is not None, 'for a decaying learning rate decay must be specified'
            try:
                lr = getattr(utils, decay)
            except TypeError:
                raise TypeError('for a decaying learning rate the decay must be specify'
                                ' as a string in [Time_Step, Step, Exponential]')

        else:
            def lr(learning_rate, *kargs):
                return learning_rate

        start_time = time.time()
        S = np.zeros(shape=(self.d, 1))
        for k in range(K):
            theta = Theta_0
            gradient = self.Loss.complete_grad(theta)
            A, S, S_ = self.approx_hessian(theta, S, k, T)
            for t in range(T):
                i = np.random.randint(low=0, high=self.N-1)
                theta -= lr(learning_rate, k, rate, t, T) * self.get_update(i, theta, Theta_0, gradient, A, S_)
                if save_at_every_step:
                    self.add_memory_entry(theta.tolist(), self.Loss.loss(i, theta), time.time()-start_time)

            Theta_0 = theta
            if not save_at_every_step:
                self.add_memory_entry(theta.tolist(), self.Loss.mean_loss(theta), time.time() - start_time)

        self.Theta = theta

    def approx_hessian(self, *kargs):
        return 0, 0, 0

    def get_update(self, *kargs):
        pass

    def add_memory_entry(self, theta, loss, t):
        self.theta_memory.append(theta)
        self.loss_memory.append(loss)
        self.time_memory.append(t)


class SGD(genral_model):
    '''
    Stochastic gradient descent
    '''
    def __init__(self):
        super().__init__()


    def get_update(self, i, theta, Theta_0, gradient, *kargs):
        return self.Loss.grad(i, theta)


class SVRG(genral_model):
    '''
    Stochastic Variance Reduction Gradient
    '''
    def __init__(self):
        super().__init__()

    def get_update(self, i, theta, Theta_0, gradient, *kargs):
        return self.Loss.grad(i, theta) - self.Loss.grad(i, Theta_0) + gradient


class SVRG2(genral_model):
    '''
    Stochastic Variance Reduction Gradient with tracking
    '''
    def __init__(self):
        super().__init__()

    def get_update(self, i, theta, Theta_0, gradient, hessian, *kargs):
        return self.Loss.grad(i, theta) - self.Loss.grad(i, Theta_0)\
               - np.dot(self.Loss.hess(i, theta), (theta - Theta_0)) + gradient + np.dot(hessian, (theta - Theta_0))

    def approx_hessian(self, theta, *kargs):
        return self.Loss.complete_hess(theta), 0, 0


class CM(genral_model):
    '''
    Curvature Matching
    '''
    def __init__(self):
        super().__init__()
        self.updates = []

    def get_update(self, i, theta, Theta_0, gradient, A, S):
        update = self.Loss.grad(i, theta) - self.Loss.grad(i, Theta_0)\
               - np.linalg.multi_dot([A, np.transpose(S), self.Loss.hess(i, theta),
                                      S, np.transpose(A), (theta - Theta_0)]) +\
               gradient + np.dot(A, np.dot(np.transpose(A), (theta - Theta_0)))

        self.updates.append(update)

        return update

    def approx_hessian(self, theta, S, k, T):
        if k == 0:
            return np.zeros(shape=(self.d, self.d)), np.zeros(shape=(self.d, self.d)), np.zeros(shape=(self.d, self.d))

        else:
            S = utils.Compute_S(self.d, k, T, self.updates)
            A_temp = self.Loss.complete_hess(theta, S=S)
            C = np.linalg.pinv(np.dot(np.transpose(S), A_temp))
            self.updates = []

            return np.dot(A_temp, C), S, np.dot(S, C)


class AM(CM):
    '''
    Curvature Matching
    '''
    def __init__(self):
        super().__init__()


    def get_update(self, i, theta, Theta_0, gradient, A, S):
        update = self.Loss.grad(i, theta) - self.Loss.grad(i, Theta_0)\
               - np.dot(np.linalg.multi_dot([A, np.transpose(S), self.Loss.hess(i, theta),
                            (np.identity(self.d) - np.dot(S, np.transpose(A)))]) +
                        np.linalg.multi_dot([self.Loss.hess(i, theta), S, np.transpose(A)]),
                        (theta - Theta_0)) +\
               gradient + np.dot(A, np.dot(np.transpose(A), (theta - Theta_0)))

        self.updates.append(update)

        return update



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.normal(size=(10000, 1000))
    Theta = np.random.normal(size=(1000,1))
    y = np.dot(X, Theta).reshape((10000,)) + np.random.normal(scale=0.1, size=(10000,))

    # svrg = SVRG2()
    # svrg.fit([y, X], K=10, T=200, learning_rate=0.001, save_at_every_step=True)
    # print(svrg.Theta)
    #
    # plt.plot(np.array(svrg.theta_memory)[:, 0])
    # plt.show()
    # plt.plot(svrg.loss_memory)
    # plt.show()

    svrg = AM()
    svrg.fit([y, X], K=10, T=200, learning_rate=0.001, save_at_every_step=True)
    print(svrg.Theta)

    plt.plot(np.array(svrg.theta_memory)[:, 0])
    plt.show()
    plt.plot(svrg.loss_memory)
    plt.show()