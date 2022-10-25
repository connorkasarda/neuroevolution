"""
Purpose: Learn about neural networks by building a simple 3-layer (input, hidden, output) neural network class from scratch.
Reference: Rashid, T. Make your own neural network Tariq Rashid. (CreateSpace Independent Publishing Platform, 2016).
"""
# Adds math packages for neural network
import numpy, scipy.special
# Blueprint for 3-layer neural network
class Network:
    # Neural Network constructor
    def __init__(self, inputs, hiddens, outputs, alpha):
        # Sets number of nodes for each layer and alpha (learning rate)
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.alpha = alpha
        # Declares lambda functions that connect nodes with weights, activation signals, verticalizations of lists, and weight updates
        self.connect = lambda i, o: numpy.random.normal(0.0, pow(i, -0.5), (o, i))
        self.activate = lambda x: scipy.special.expit(x)
        self.verticalize = lambda a: numpy.array(a, ndmin=2).T
        self.update = lambda e, o, i: self.alpha * numpy.dot((e * o * (1.0 - o)), numpy.transpose(i))
        # Initialize weight matrices to connect the gaps between each pair of layers
        self.innerweb = self.connect(inputs, hiddens)
        self.outerweb = self.connect(hiddens, outputs)
    # Performs forward propagation to query the network
    def query(self, inputs):
        # Converts inputs list into a vertical list
        inputactivation = self.verticalize(inputs)
        # Calculates hidden layer output
        hiddensignals = numpy.dot(self.innerweb, inputactivation)
        hiddenactivation = self.activate(hiddensignals)
        # Calculates final output of neural network
        outputsignals = numpy.dot(self.outerweb, hiddenactivation)
        outputactivation = self.activate(outputsignals)
        # Returns hidden and output activation
        return hiddenactivation, outputactivation
    # Performs backward propagation to train the network
    def train(self, inputs, targets):
        # Computes a forward query on neural network
        hiddenactivation, outputactivation = self.query(inputs)
        # Converts inputs and targets lists to vertical lists
        inputactivation = self.verticalize(inputs)
        target = self.verticalize(targets)
        # Calculates hidden and output errors
        outputerrors = target - outputactivation
        hiddenerrors = numpy.dot(self.outerweb.T, outputerrors)
        # Updates inner and outer weights
        self.outerweb += self.update(outputerrors, outputactivation, hiddenactivation)
        self.innerweb += self.update(hiddenerrors, hiddenactivation, inputactivation)