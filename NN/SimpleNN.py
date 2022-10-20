"""
Builds and tests a simple neural network.
"""
# Adds math packages for neural network
import numpy
# Blueprint for n-layer neural network
class NeuralNetwork:
    # Neural Network constructor
    def __init__(self, nodes, learnrate) -> None:
        # Sets the list that represents number of nodes per layer as well as learning rate
        self.nodes = nodes
        self.learnrate = learnrate
        # Creates randomly weighted links between each pair of layers based on the Gaussian distribution
        self.links = [numpy.random.normal(0.0, pow(self.nodes[layer - 1], -0.5), (self.nodes[layer], self.nodes[layer - 1])) for layer in range(1, len(self.nodes))]
        # Adds a lambda function that computes the sigmoid activation function
        self.activatefunc = lambda x: 1 / (1 + numpy.exp(-x))
    # Performs forward propagation to query the network
    def forward(self, inputs):
        # Convert horizontal inputs list into a vertical list and use to initialize outputs
        outputs = numpy.array(inputs, ndmin=2).T
        # Propogate over each matrix of links, computing signals
        for layer in range(1, len(self.nodes)):
            inputs = outputs
            recieved = numpy.dot(self.links[layer - 1], inputs)
            outputs = self.activatefunc(recieved)
        # Return the prediction
        return outputs
    # Performs backward propagation to train the network
    def backward(self):
        pass