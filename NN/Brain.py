"""
Purpose: Simple 3-layer (input, hidden, output) neural network class.
Reference: Rashid, T. Make your own neural network Tariq Rashid. (CreateSpace Independent Publishing Platform, 2016).
"""
# Adds math packages for neural network
import numpy, scipy.special
# Blueprint for 3-layer neural network
class Brain:
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
        inputsignals = self.verticalize(inputs)
        # Calculates hidden and outer level signals
        hiddensignals = numpy.dot(self.innerweb, inputsignals)
        hiddenvalues = self.activate(hiddensignals)
        # Sends signals through activation function for final values
        outputsignals = numpy.dot(self.outerweb, hiddenvalues)
        outputvalues = self.activate(outputsignals)
        # Returns hidden and output values
        return hiddenvalues, outputvalues
    # Performs backward propagation to train the network
    def train(self, inputs, targets):
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

# Instantiates NN object
numinputs = 784
numhiddens = 200
numoutputs = 10
learnrate = 0.1
print('STATUS: EXPERIMENT STARTED ...')
brain = Brain(numinputs, numhiddens, numoutputs, learnrate)
# Loads the training file
trainfile = open('NN/mnist_train.csv', 'r')
trainlist = trainfile.readlines()
trainfile.close()
# Begins training
print('STATUS: TRAINING ...')
epochs = 5
# Trains neural network multiple times
for epoch in range(epochs):
    print(' STATUS: EPOCH', epoch + 1, 'of', epochs, '...')
    # Iterates over each record in the training dataset
    for rec in trainlist:
        # Splits the records
        data = rec.split(',')
        # Scales the input values
        inputs = (numpy.asfarray(data[1:]) / 255.0 * 0.99) + 0.01
        # Sets up the target values
        targets = numpy.zeros(numoutputs) + 0.01
        targets[int(data[0])] = 0.99
        # Trains the neural network
        brain.train(inputs, targets)
# Loads in the testing file
testfile = open('NN/mnist_test.csv', 'r')
testlist = testfile.readlines()
testfile.close()
# Setups variables to keep track of total tests and score
score = []
# Iterates over each record in the testing dataset
print('STATUS: TESTING ...')
for rec in trainlist:
    # Splits the records
    data = rec.split(',')
    # Scales the input values
    inputs = (numpy.asfarray(data[1:]) / 255.0 * 0.99) + 0.01
    # Retrieves the label
    label = int(data[0])
    # Queries the neural network
    hiddens, outputs = brain.query(inputs)
    # Retrieves the prediction
    predicted = numpy.argmax(outputs)
    # Increment score counter only if correct prediction made
    score.append(1) if predicted == label else score.append(0)
# Prints results of the test
performance = numpy.asarray(score)
print('SCORE: ', performance.sum() / performance.size)
print('STATUS: EXPERIMENT COMPLETE!')