"""
Builds and tests a simple 3-layer (input, hidden, output) neural network.
"""
# Adds math packages for neural network
import numpy
# Blueprint for 3-layer neural network
class NN:
    # Neural Network constructor
    def __init__(self, inputs: int, hiddens: int, outputs: int, learnrate: float) -> None:
        # Sets number of nodes for each layer and learning rate
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.learnrate = learnrate
        # Declares lambda functions that compute weight matrices, activation signals, verticalizations of lists, and weight updates
        self.connect = lambda x, y: numpy.random.normal(scale=pow(x, -0.5), size=(y, x))
        self.activation = lambda x: 1 / (1 + numpy.exp(-x))
        self.verticalize = lambda x: numpy.array(x, ndmin=2).T
        self.update = lambda x, y, z: self.learnrate * numpy.dot((x * y * (1.0 - y)), numpy.transpose(z))
        # Initialize weight matrices to connect the gaps between each pair of layers
        self.innerweb = self.connect(inputs, hiddens)
        self.outerweb = self.connect(hiddens, outputs)
    # Performs forward propagation to query the network
    def query(self, inputs: list[float]) -> tuple:
        # Converts inputs list into a vertical list
        inputvalues = self.verticalize(inputs)
        # Calculates hidden and outer level signal values
        hiddenvalues = self.activation(numpy.dot(self.innerweb, inputvalues))
        outputvalues = self.activation(numpy.dot(self.outerweb, hiddenvalues))
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