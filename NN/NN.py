"""
Builds and tests a simple 3-layer (input, hidden, output) neural network.
"""
# Adds math packages for neural network
import numpy
# Blueprint for 3-layer neural network
class NN:
    # Neural Network constructor
    def __init__(self, inputs: int, hiddens: int, outputs: int, alpha: float) -> None:
        # Sets number of nodes for each layer and alpha (learning rate)
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.alpha = alpha
        # Declares lambda functions that connect nodes with weights, activation signals, verticalizations of lists, and weight updates
        self.connect = lambda i, o: numpy.random.normal(scale=pow(i, -0.5), size=(o, i))
        self.activate = lambda x: 1 / (1 + numpy.exp(-x))
        self.verticalize = lambda a: numpy.array(a, ndmin=2).T
        self.update = lambda e, o, i: self.alpha * numpy.dot((e * o * (1.0 - o)), numpy.transpose(i))
        # Initialize weight matrices to connect the gaps between each pair of layers
        self.innerweb = self.connect(inputs, hiddens)
        self.outerweb = self.connect(hiddens, outputs)
    # Performs forward propagation to query the network
    def query(self, inputs: list[float]) -> tuple:
        # Converts inputs list into a vertical list
        inputvalues = self.verticalize(inputs)
        # Calculates hidden and outer level signal values
        hiddenvalues = self.activate(numpy.dot(self.innerweb, inputvalues))
        outputvalues = self.activate(numpy.dot(self.outerweb, hiddenvalues))
        # Returns hidden and output values
        return hiddenvalues, outputvalues
    # Performs backward propagation to train the network
    def train(self, inputs: list[float], targets: list[float]) -> None:
        # Computes a forward query on neural network
        hiddenvalues, outputvalues = self.query(inputs)
        # Converts inputs and targets lists to vertical lists
        inputvalues = self.verticalize(inputs)
        targetvalues = self.verticalize(targets)
        # Calculates hidden and output errors
        outputerrors = targetvalues - outputvalues
        hiddenerrors = numpy.dot(self.outerweb.T, outputerrors)
        # Updates inner and outer weights
        self.outerweb += self.update(outputerrors, outputvalues, hiddenvalues)
        self.innerweb += self.update(hiddenerrors, hiddenvalues, inputvalues)