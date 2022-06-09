import pickle
import analysis
import numpy as np

''' An implementation of an MLP with a single layer of hidden units. '''


class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.
        
        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.
        
        Note: a1 and z1 can be used for caching during backprop/evaluation.
        
        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units

        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2 * (np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2 * (np.random.random((self.dout, self.hidden_units)) - 0.5)
        self.z1 = np.zeros(self.hidden_units)
        self.a1 = np.zeros(self.hidden_units)

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def eval(self, xdata):

        N = np.size(xdata, 1)

        yout = np.zeros((self.dout, N))
        for i in range(N):  # for all the data points
            X = xdata[:, i]

            Y = np.zeros(self.dout)
            for j in range(self.dout):  # for all the network outputs
                final = self.b2[j]
                self.z1 = np.zeros(self.hidden_units)
                self.a1 = np.zeros(self.hidden_units)
                for n in range(self.hidden_units):  # for all the hidden units

                    total = self.b1[n]

                    for m in range(self.din):  # for all the inputs of a data point

                        total = total + X[m] * self.W1[n, m]

                    final = final + self.W2[j, n] * np.tanh(total)
                    self.z1[n] = np.tanh(total)
                    self.a1[n] = total

                Y[j] = final

            expsum = 0.0

            for k in range(len(Y)):
                # if np.exp(Y[k]) > 30000000:
                #   print(Y[k])
                expsum = expsum + np.exp(Y[k])

            for h in range(len(Y)):
                yout[h, i] = np.exp(Y[h]) / expsum

        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''
        return yout

    def sgd_step(self, xdata, ydata, learn_rate):

        N = np.size(xdata, 1)

        for i in range(N):
            derivatives = self.grad(np.hsplit(xdata, N)[i], np.hsplit(ydata, N)[i])

            self.W1 = self.W1 - (learn_rate * derivatives[0])

            self.b1 = self.b1 - (learn_rate * derivatives[1])
            # print(self.W2)
            self.W2 = self.W2 - (learn_rate * derivatives[2])

            self.b2 = self.b2 - (learn_rate * derivatives[3])

        ''' Do one step of SGD on xdata/ydata with given learning rate. '''
        pass

    def grad(self, xdata, ydata):

        yout = self.eval(xdata)[:, 0]

        # print("result is :")
        # print(yout)
        # print("a1 is:")
        # print(self.a1)
        # print("z1 is:")
        # print(self.z1)
        ea2 = np.zeros(len(ydata))

        ea1 = np.zeros(self.hidden_units)
        eb = 0.0

        for k in range(self.dout):
            ea2[k] = yout[k] - ydata[k, 0]
        for j in range(self.hidden_units):  # d/dx(tanh(x)) = 1 - tanh^2(x)

            scale = 1 - np.power(np.tanh(self.a1[j]), 2)

            total = 0.0
            for k in range(self.dout):
                total = total + self.W2[k, j] * ea2[k]
            ea1[j] = total * scale

        total = 0.0
        for k in range(self.dout):
            total = total + self.b2[k] * ea2[k]
        eb = total

        dW1 = np.zeros(self.W1.shape)
        dW2 = np.zeros(self.W2.shape)
        db1 = np.zeros(self.b1.shape)
        db2 = np.zeros(self.b2.shape)

        for j in range(len(self.W1)):
            for i in range(len(self.W1[0])):
                dW1[j, i] = ea1[j] * xdata[i]
        for j in range(len(self.W2)):
            for i in range(len(self.W2[0])):
                dW2[j, i] = ea2[j] * self.z1[i]

        for i in range(len(self.b1)):
            db1[i] = ea1[i]

        for j in range(len(self.b2)):
            db2[j] = ea2[j]

        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)
        
        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        tup = (dW1, db1, dW2, db2)

        return tup
