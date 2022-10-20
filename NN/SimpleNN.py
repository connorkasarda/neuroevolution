"""
Builds 3-layer neural network.
"""
# Adds math packages for neural network
import numpy
# Blueprint for simple 3-layer neural network
class SimpleNN:
    # Neural Network constructor
    def __init__(self, numinner, numhidden, numouter, learnrate) -> None:
        # Sets number of inner, hidden, and outer nodes as well as learning rate
        self.numinner = numinner
        self.numhidden = numhidden
        self.numouter = numouter
        self.learnrate = learnrate
        # Creates randomly weighted links between layers of nodes based on the Gaussian distribution
        self.innerlinks = numpy.random.normal(0.0, pow(self.numinner, -0.5), (self.numhidden, self.numinner))
        self.outerlinks = numpy.random.normal(0.0, pow(self.numouter, -0.5), (self.numouter, self.numhidden))
        # Adds a lambda function that computes the activation function
        self.activatefunc = lambda x: 1 / (1 + numpy.exp(-x))