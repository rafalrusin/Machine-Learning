import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.seterr(all = 'ignore')

# transfer functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

# using tanh over logistic sigmoid for the hidden layer is recommended   
def tanh(x):
    return np.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y

class MLP_Classifier(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm can be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """
    def __init__(self, input, hidden, output, iterations = 50, learning_rate = 0.01, 
                l2_in = 0, l2_out = 0, momentum = 0, rate_decay = 0, 
                output_layer = 'logistic', verbose = True):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param l2: L2 regularization term
        :param momentum: momentum
        :param rate_decay: how much to decrease learning rate by on each iteration (epoch)
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to spit out error rates while training
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.verbose = verbose
        self.output_activation = output_layer
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden 
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            print(len(inputs))
            print(self.input-1)
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        self.ai[0:self.input -1] = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = sigmoid(sum)
        self.ah[self.hidden - 1] = 1. # Hidden bias
            
        # output activations
        sum = np.dot(self.wo.T, self.ah)
        if self.output_activation == 'logistic':
            self.ao = sigmoid(sum)
        elif self.output_activation == 'softmax':
            self.ao = softmax(sum)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p') 
        
        
        return self.ao

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative

        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta (theta) tell you which direction to change the weights
        if self.output_activation == 'logistic':
            output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        elif self.output_activation == 'softmax':
            output_deltas = -(targets - self.ao)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p') 

        #print("Debug")
        #print(self.ai)
        #print(self.ah)
        #print(self.ao)
        #print(targets)
        #print(output_deltas)
        
        # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dsigmoid(self.ah) * error
            
        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum 
        self.co = change 

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum 
        self.ci = change

        # calculate error
        if self.output_activation == 'softmax':
            error = -sum(targets * np.log(self.ao))
        elif self.output_activation == 'logistic':
            error = sum(0.5 * (targets - self.ao)**2)
        
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def fit(self, patterns):
        if self.verbose == True:
            if self.output_activation == 'softmax':
                print('Using softmax activation in output layer')
            elif self.output_activation == 'logistic':
                print('Using logistic sigmoid activation in output layer')
                
        num_example = np.shape(patterns)[0]
                
        for i in range(self.iterations):
            error = 0.0
            patternOrder = list(range(len(patterns)))
            random.shuffle(patternOrder)
            for j in patternOrder:
                p = patterns[j]
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
                
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
                
            if i % 10 == 0 and self.verbose == True:
                error = error/num_example
                #print('Training error %i %-.5f' % (i, error))
                
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

def demo():
    from sklearn.preprocessing import scale
    """
    run NN demo on the digit recognition dataset from sklearn
    """
    def load_data():
        data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')

        # first ten values are the one hot encoded y (target) values
        y = data[:,0:10]
        
        data = data[:,10:] # x data
        data = scale(data)
        
        out = []
        testout = []
        #print data.shape

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            tupledata = list((data[i,:].tolist(), y[i].tolist())) # don't mind this variable name
            
            if ((i//10)%2) == 0:
                out.append(tupledata)
            else:
                testout.append(tupledata)

        return out, testout

    def load_data2(fn):
        out = []
        testout = []
        i = 0
        for x in np.arange(0, 1, 0.001):
            tupledata = [[x], [fn(x)]]
            if (i%2) == 0:
                out.append(tupledata)
            else:
                testout.append(tupledata)
            i = i + 1

        return out, testout
        
    def testfn1(x):
        return np.sin((x - 0.3) * np.pi * 3) / 4 + 0.5
    
    start = time.time()
    
    X,T = load_data2(testfn1)

    print(X[9]) # make sure the data looks right
    print(T[2]) # make sure the data looks right

    NN = MLP_Classifier(1, 60, 1, iterations = 10, learning_rate = 0.01, 
                        momentum = 0.5, rate_decay = 0.0000, 
                        output_layer = 'logistic', verbose = False)

    NN.fit(X)
    
    end = time.time()
    print(end - start)
    print("finished")
    
    #NN.test(X)

    overallErrorSqr, overallErrorN = 0, 0
    
    for t in T:
        p = NN.predict([t[0]])
        #print("t[1]: " + str(t[1]) + ", p: " + str(p[0]))
        
        errorSqr, errorN = 0, 0
        for j in range(0, 1):
            #print(j)
            errorSqr = errorSqr + (p[0][j] - t[1][j]) ** 2
            errorN = errorN + 1

        error = errorSqr ** 0.5 / errorN
        #print("Error for p: " + str(p[0]) + ", t: " + str(t[1]) + " = " + str(error))

        overallErrorSqr = overallErrorSqr + error ** 2
        overallErrorN = overallErrorN + 1

    overallError = overallErrorSqr ** 0.5 / overallErrorN
    print("overallError: " + str(overallError))
    if overallError > 0.001:
        print("Too big error")

    fig, ax = plt.subplots()

    x = []
    y = []
    z = []
    for t in T:
        x.append(t[0])
        y.append(t[1])
        z.append(NN.predict([t[0]])[0])
    ax.plot(x, y, 'r--')
    line, = ax.plot(x, z)

    def animate(v):
        print(v)

        NN.fit(X)
        
        z = []
        for t in T:
            z.append(NN.predict([t[0]])[0])
            
        line.set_ydata(z)
        return line,

    def init():
        return line,
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, 2000), interval=25, init_func=init, blit=True)
    plt.show()
        

if __name__ == '__main__':
    demo()
