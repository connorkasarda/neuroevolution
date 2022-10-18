"""
Computes 3-layer neural network.
Code comes from Make Your Own Neural Network book mentioned in References of README.
This was used to help learn more about how neural networks work.
Will use this as a foundation for learning more about neural networks, especially n-layered neural networks and NeuroEvolution concepts.
"""
# Import necessary math packages
import numpy, scipy.special, matplotlib.pyplot
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

# Sets number of input, hidden, and output nodes for neural network object
num_inputs = 784
num_hiddens = 100
num_outputs = 10
# Sets learning rate
learn_rate = 0.3
# Creates the instance of neural network
nn = SimpleNN(num_inputs, num_hiddens, num_outputs, learn_rate)
# Loads the MNIST training dataset (IMPORTANT: download the MNIST train and test dataset yourself! Otherwise, won't work!)
train_file = open('NN/mnist_train.csv', 'r')
train_list = train_file.readlines()
train_file.close()
# Goes through each record in training dataset
for rec in train_list:
    # Splits data with commas
    values = rec.split(',')
    # Scale the input data
    inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
    # Creates target output values where final outputs are sent to
    targets = numpy.zeros(num_outputs) + 0.01
    # Emphasize that zeroeth elements are just labels
    targets[int(values[0])] = 0.99
    # Trains the neural network
    nn.train(inputs, targets)
# Loads the MNIST testing dataset
test_file = open('NN/mnist_test.csv', 'r')
test_list = test_file.readlines()
test_file.close()
# Sets the scorecard to keep track of neural network performance
scorecard = []
# Goes through each record in testing dataset
for rec in test_list:
    # Separates values by commas
    values = rec.split(',')
    # Stores correct value
    correct = int(values[0])
    # Scales inputs
    inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
    # Queries the neural network
    outputs = nn.query(inputs)
    # Retrieve the predicted value
    label = numpy.argmax(outputs)
    # Choose if predicted is correct or incorrect and add to scorecard
    if (label == correct): scorecard.append(1)
    else: scorecard.append(0)
# Calculates and prints the performance of the neural network
array = numpy.asarray(scorecard)
print('Performance: ', array.sum() / array.size)