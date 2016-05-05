import numpy as np
import sys
import math

class NeuralNetwork:

    # Initialize Neural Network
    def __init__(self, neurons_per_layer):  # For instance, neurons_per_layer = [2,3,2]
        self.neurons_per_layer = neurons_per_layer  # Neurons in each layer
        self.num_layers = len(neurons_per_layer)  # Number of layers
        self.neuron_layers_list = []  # Vector containing all Neuron Layer objects
        self.init_neuron_layers()

    # Initialize all neuron layers
    def init_neuron_layers(self):

        # Loop for all layers and assign them their corresponding number of inputs and neurons
        for layer_index in range(1, self.num_layers):
            layer = NeuronLayer(self.neurons_per_layer[layer_index - 1],
                                self.neurons_per_layer[layer_index], layer_index)  # Create new neuron layer
            self.add_layer(layer)  # Add the new neuron layer to the list of neuron layers

    # Add neuron layer to the list of neuron layers
    def add_layer(self, layer):
        self.neuron_layers_list.append(layer)

    # Check if Layers have been correctly initialized
    def check(self):
        print '\n-----------------------------------------'
        print 'Neural Network details: Layer transitions'
        print '-----------------------------------------\n'

        for neuron_layer in self.neuron_layers_list:
            print neuron_layer.num_inputs, '->', neuron_layer.num_neurons
            print 'Activation = ', neuron_layer.activations
            print 'Activity = ', neuron_layer.activities

    # Execute forward algorithm
    def forward(self, feature_vector):
        inputs = feature_vector
        for neuron_layer in self.neuron_layers_list:
            inputs = neuron_layer.forward(inputs)

    def train(self):
        a = 0


class NeuronLayer:

    # Initialize the neuron layer
    def __init__(self, num_inputs, num_neurons, ID):
        self.ID = ID  # ID of the layer
        self.num_inputs = num_inputs  # Number of inputs
        self.num_neurons = num_neurons  # Number of neurons in the layer
        self.weight_matrix = np.matrix  # Weight matrix of this layer
        self.init_weights()
        self.activations = np.matrix([])  # Activations (all neurons) in this layer
        self.activities = np.array([])  # Activity (all neurons) in this layer, i.e. tresholded activations

    # Initialize the weights of this layer, i.e. the weight matrix W
    def init_weights(self):
        # Mean and standard deviation of the initialization of the weight matrix
        mu = 0
        sigma = 0.5
        self.weight_matrix = np.random.normal(mu, sigma, [self.num_neurons, self.num_inputs])

    # Compute activation and activity values in this layer
    def forward(self, inputs):
        # Compute activations
        try:
            self.activations = self.weight_matrix.dot(inputs)
        except:
            print >> sys.stderr, 'Error computing activation at layer', self.ID,'-- Please check the input feature vectors.'
            quit()
        # Compute activity
        self.non_linearity()
        return self.activities

    # Apply non-linearity to the activations
    def non_linearity(self):
        for activation in self.activations:
            self.activities = np.append(self.activities, sigmoid(activation))

# Sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))


v = np.array([0.05, 0.10])
a = NeuralNetwork((2, 2, 2, 2))
a.forward(v)
a.check()
