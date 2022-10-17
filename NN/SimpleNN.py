# Import necessary math packages
import numpy, scipy.special
# Constructs 3-layer (input, hidden, and output) neural network from scratch using OOP
class SimpleNN:
    # Initializes neural network
    def __init__(self, inputs, hiddens, outputs, lrate):
        # Sets number of input, hidden, and output nodes
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        # Links layers of nodes with initial, random-weighted connections
        self.ihweights = numpy.random.normal(0.0, pow(self.inputs, -0.5), (self.hiddens, self.inputs))
        self.howeights = numpy.random.normal(0.0, pow(self.hiddens, -0.5), (self.outputs, self.hiddens))
        # Sets the learning rate
        self.lrate = lrate
        # Sets the activation function to the sigmoid function
        self.afunc = lambda x: scipy.special.expit(x)
    # Trains the neural network
    def train(self, ilist, tlist):
        # Converts input and target lists to 2D arrays
        isignals = numpy.array(ilist, ndmin=2).T
        tsignals = numpy.array(tlist, ndmin=2).T
        # Computes signal values into hidden layer
        hsignals = self.afunc(numpy.dot(self.ihweights, isignals))
        # Computes signal values into output layer
        osignals = self.afunc(numpy.dot(self.howeights, hsignals))
        # Computes output layer signal error
        oerrors = tsignals - osignals
        # Computes hidden layer signal error
        herrors = numpy.dot(self.howeights.T, oerrors)
        # Updates the hidden-output weighted links based on computed errors
        self.howeights += self.lrate * numpy.dot((oerrors * osignals * (1.0 - osignals)), numpy.transpose(hsignals))
        # Updates the input-hidden weighted links based on computed errors
        self.ihweights += self.lrate * numpy.dot((herrors * hsignals * (1.0 - hsignals)), numpy.transpose(isignals))
    # Queries the neural network
    def query(self, ilist):
        # Converts input list to 2D array
        isignals = numpy.array(ilist, ndmin=2).T
        # Computes signal values into hidden layer
        hsignals = self.afunc(numpy.dot(self.ihweights, isignals))
        # Computes signal values into output layer
        osignals = self.afunc(numpy.dot(self.howeights, hsignals))
        # Returns the output signals as final output
        return osignals