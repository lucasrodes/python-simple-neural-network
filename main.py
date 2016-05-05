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
        self.neuron_layers_list[self.num_layers-2].output_layer = True # Set flag "output_layer" to True for last NL

    # Add neuron layer to the list of neuron layers
    def add_layer(self, layer):
        self.neuron_layers_list.append(layer)

    # Check if Layers have been correctly initialized
    def check(self):
        print '\n-----------------------------------------'
        print 'Neural Network details: Layer transitions'
        print '-----------------------------------------\n'
        count = 0
        for neuron_layer in self.neuron_layers_list:
            print 'Transition', neuron_layer.ID, ':', neuron_layer.num_inputs, '->', neuron_layer.num_neurons
            print 'Weight Matrix =', neuron_layer.weight_matrix
            if count == self.num_layers-2:
                print 'last'
                break
            else:
                print 'Activation = ', neuron_layer.activations
                print 'Activity = ', neuron_layer.activities
            print '\n-----------------------------------------------------'
            count += 1
        print
        print 'Estimated Output = ', neuron_layer.activations

    # Execute forward algorithm
    def forward(self, feature_vector):
        inputs = feature_vector
        # Flip neuron layer list, since error BACK-propagates
        neuron_layers_list = self.neuron_layers_list[::-1]
        for neuron_layer in neuron_layers_list:
            inputs = neuron_layer.forward(inputs)
        return inputs  # This is the estimated output vector

    # Execute the backward algorithm
    def backward(self, output_error_gradient):
        delta = output_error_gradient
        for neuron_layer in self.neuron_layers_list:
            delta = neuron_layer.backward(delta)

    def train(self, feature_vector, real_output):
        estimated_output = self.forward(feature_vector)
        output_error_gradient = -2*(real_output - estimated_output)  # Squared Error: E = 1/2*(real-estimation)^2
        self.backward(output_error_gradient)


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
        self.output_layer = False  # Flag to denote if this layer is the output layer

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
        # Compute activities
        self.activation_function()
        return self.activities

    # Apply non-linearity to the activations
    def activation_function(self):
        if self.output_layer:
            self.activities = self.activations
        else:
            for activation in self.activations:
                self.activities = np.append(self.activities, sigmoid(activation))


# Sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))


v = np.array([0.05, 0.10])
a = NeuralNetwork((2, 2, 2, 2))
a.forward(v)

print 'Input = ', v
a.check()
