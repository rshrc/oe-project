import numpy as np
import matplotlib.pyplot as plt



class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 128

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize)
        self.W3 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        self.b1 = np.random.randn(1, self.hiddenLayerSize)
        self.b2 = np.random.randn(1, self.hiddenLayerSize)
        self.b3 = np.random.randn(1, self.outputLayerSize)

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3) + self.b3
        yHat = self.z4
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J[0]

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        # don't get sigmoid prime in output layer.
        # we don't use activation function in output layer
        delta4 = -(y - self.yHat)
        dJdW3 = np.dot(self.a3.T, delta4)
        dJdb3 = np.mean(delta4, axis=0)

        # after then, we need sigmoid prime because of activate function(sigmoid)
        delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdb2 = np.mean(delta3, axis=0)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        dJdb1 = np.mean(delta2, axis=0)

        return dJdW1, dJdW2, dJdW3, dJdb1, dJdb2, dJdb3

    def gradient_descent(self, lr, dJdW1, dJdW2, dJdW3, dJdb1, dJdb2, dJdb3):

        self.W1 = self.W1 - lr * dJdW1
        self.W2 = self.W2 - lr * dJdW2
        self.W3 = self.W3 - lr * dJdW3

        self.b1 = self.b1 - lr * dJdb1
        self.b2 = self.b2 - lr * dJdb2
        self.b3 = self.b3 - lr * dJdb3

    # not necessary but worthy to reconsider what is hyper parameter
    def opt_hyper_params(self, X, y):
        best_cost = 100000
        best_params = {'input_dim': None, 'hidden_dim': None}

        for dim_in in range(1, 30):
            for dim_hid in range(128, 527, 40):
                self.inputLayerSize = dim_in
                self.hiddenLayerSize = dim_hid

                cost = self.costFunction(X, y)

                if cost < best_cost:
                    best_params['input_dim'] = dim_in
                    best_params['hidden_dim'] = dim_hid
                    best_cost = cost

        self.inputLayerSize = best_params['input_dim']
        self.hiddenLayerSize = best_params['hidden_dim']


def train(epoch, lr, X, y):
    NN = Neural_Network()
    print('Neaural Network built.')

    # not necessary
    NN.opt_hyper_params(X, y)
    print('\nHyper parameter optimization is done.')
    print(' Input layer size: {} \t Hidden layer size: {}'.format(NN.inputLayerSize, NN.hiddenLayerSize))

    print('\nTraining ANN...')


    costs = []  # list of cost for each epoch
    # start training
    for epoch in range(epoch + 1):
        # calculate gradients
        dJdW1, dJdW2, dJdW3, dJdb1, dJdb2, dJdb3 = NN.costFunctionPrime(X, y)

        # update Weight with gradient
        NN.gradient_descent(lr, dJdW1, dJdW2, dJdW3, dJdb1, dJdb2, dJdb3)

        cost = NN.costFunction(X, y)
        costs.append(cost)
        print("epoch: {}, cost: {}".format(epoch, cost))

    # plot train process, showing cost of each epoch
    x_axis = np.arange(0, epoch + 1)
    plt.plot(x_axis, costs)
    plt.xlabel = 'epoch'
    plt.ylabel = 'cost'
    plt.grid(1)
    plt.show()


###################
# 4. generate data
###################
# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)



#################
# 5.training ANN
#################

train(50, 0.001, X, y)

