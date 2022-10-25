"""
Purpose: Learn about deep learning by building an artificial neural network (can have more than 1 hidden layer) from scratch.
         Instead of layers of nodes, represented as layers of "webs" or matrices
"""
# Adds math packages for neural network
import numpy, scipy.special
# Blueprint for each layer in the neural network
class Web:
    # Web matrix constructor, where parents are nodes from previous layer and children are nodes from current layer
    def __init__(self, parents, children, alpha):
        # Sets number of previous layer, current layer nodes, and learning rate
        self.parents = parents
        self.children = children
        self.alpha = alpha
        # Connects the two layers with a matrix of weighted links
        self.threads = numpy.random.normal(0.0, pow(parents, -0.5), (children, parents))
    # Computes a forward of the web, likely for a query
    def send(self, inputs):
        # Calculates output of the web
        signals = numpy.dot(self.threads, inputs)
        activations = scipy.special.expit(signals)
        # Returns the output of the web
        return activations
    def update(self, errors, outers, inners):
        # Calculates change in weights
        delta = self.alpha * numpy.dot((errors * outers * (1.0 - outers)), numpy.transpose(inners))
        self.threads += delta
# Blueprint for n-layer neural network
class Artificial:
    # ANN Constructor
    def __init__(self, nodes, alpha):
        # Sets number of nodes for each layer (each index in list is a layer), and alpha (learning rate)
        self.nodes = nodes
        self.alpha = alpha
        # Webs are declared for each gap within neural network
        self.webs = [Web(self.nodes[layer], self.nodes[layer + 1], self.alpha) for layer in range(len(self.nodes) - 1)]
        # Sets up lambda functions
        self.verticalize = lambda a: numpy.array(a, ndmin=2).T
    # Performs forward propagation for prediction
    def query(self, inputs):
        # Converts inputs list into a vertical list
        data = self.verticalize(inputs)
        # Begins list of predictions with input
        predictions = []
        predictions.append(data)
        # Loops over each web until output layer is reached
        for web in self.webs:
            data = web.send(data)
            predictions.append(data)
        # Returns final output as result of query
        return predictions
    # Similar to query function, but only used for return output layer prediction
    def predict(self, inputs):
        # Converts inputs list into a vertical list
        prediction = self.verticalize(inputs)
        # Loops over each web until output layer is reached
        for web in self.webs:
            prediction = web.send(prediction)
        # Returns final output as result of query
        return prediction
    # Performs backward propagation for weight adjustments
    def train(self, inputs, targets):
        pass