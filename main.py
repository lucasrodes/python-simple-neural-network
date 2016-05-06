import numpy as np
import sys
import math


class NeuralNetwork:
    #LEARNING_RATE = 0.1

    # Initialize Neural Network
    def __init__(self, neurons_per_layer):  # For instance, neurons_per_layer = [2,3,2]
        self.neurons_per_layer = neurons_per_layer  # Neurons in each layer
        self.num_layers = len(neurons_per_layer)  # Number of layers
        self.neuron_layers_list = []  # Vector containing all Neuron Layer objects
        self.LEARNING_RATE = 0.2
        self.init_neuron_layers()

    # Initialize all neuron layers
    def init_neuron_layers(self):

        # Loop for all layers and assign them their corresponding number of inputs and neurons
        for layer_index in range(1, self.num_layers):
            layer = NeuronLayer(self.neurons_per_layer[layer_index - 1],
                                self.neurons_per_layer[layer_index], layer_index, self.LEARNING_RATE)  # Create new neuron layer
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
                break
            else:
                print 'Activation = ', neuron_layer.activations
                print 'Activity = ', neuron_layer.activities
            print '\n-----------------------------------------------------'
            count += 1
        print
        print 'Estimated Output = ', neuron_layer.activities

    # Execute forward algorithm
    def forward_step(self, feature_vector):
        inputs = feature_vector
        for neuron_layer in self.neuron_layers_list:
            inputs = neuron_layer.forward_step(inputs)  # activity of layer L is input of layer L+1
        return inputs  # This is the estimated output vector

    # Execute the backward algorithm
    def backward_step(self, output_error_gradient):
        delta = output_error_gradient
        weight_matrix = None
        # Flip neuron layer list, since error BACK-propagates
        neuron_layers_list = self.neuron_layers_list[::-1]
        for neuron_layer in neuron_layers_list:
            delta, weight_matrix = neuron_layer.backward_step(delta, weight_matrix)

    def train(self, feature_vector, real_output):
        for epoc in range(1, 1000):
            if epoc % 100 == 0:
                print 'Epoc ', epoc
            #print len(feature_vector)
            for i in range(0, len(feature_vector)):
                estimated_output = self.forward_step(feature_vector[i])
                #print 'Est = ', estimated_output, 'Real = ', real_output[i], 'Dif = ', estimated_output-real_output[i]
                output_error_gradient = -2*(real_output[i] - estimated_output)  # Squared Error: E = 1/2*(real-estimation)^2
                self.backward_step(output_error_gradient)
            self.update_learning_rate(0.99)

    def update_learning_rate(self, factor):
        for neuron_layer in self.neuron_layers_list:
            neuron_layer.LEARNING_RATE *= factor

    def input(self, feature_vector):
        return self.forward_step(feature_vector)

class NeuronLayer:

    # Initialize the neuron layer
    def __init__(self, num_inputs, num_neurons, ID, LEARNING_RATE):
        self.LEARNING_RATE = LEARNING_RATE
        self.ID = ID  # ID of the layer
        self.num_inputs = num_inputs  # Number of inputs
        self.num_neurons = num_neurons  # Number of neurons in the layer
        self.weight_matrix = np.matrix  # Weight matrix of this layer
        self.init_weights()
        self.inputs = np.array([])  # Input to this layer, i.e. activity of the previous layer
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
    def forward_step(self, inputs):
        self.inputs = inputs
        # Compute activations
        try:
            self.activations = self.weight_matrix.dot(self.inputs)
        except:
            print >> sys.stderr, 'Error computing activation at layer', self.ID,'-- Please check the input feature vectors.'
            quit()
        # Compute activities
        self.activation_function()
        return self.activities

    def backward_step(self, delta, weight_matrix_upplayer):
        # Update weights of this layer
        if not self.output_layer:
            delta = self.activities * (1 - self.activities) * np.array(delta.dot(weight_matrix_upplayer))
        self.update_weights(delta)
        return delta, self.weight_matrix

    def update_weights(self, delta):
        # print 'd= ', delta
        # print 'z=', self.inputs
        # print 'dz = ', np.outer(delta, self.inputs)
        # print 'W = ', self.weight_matrix
        self.weight_matrix -= self.LEARNING_RATE * np.outer(delta, self.inputs)


    # Apply non-linearity to the activations
    def activation_function(self):
        self.activities = np.array([])
        if self.output_layer:
            self.activities = self.activations
        else:
            for activation in self.activations:
                self.activities = np.append(self.activities, sigmoid(activation))  # Add values to the vector of actvts


# Sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Choose example
Example = 1

if Example == 1:
    N = 2
    Input = []
    Output = []
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            Input.append([i, j])
            Output.append(i+j)
    Input = np.array(Input)
    Output = np.array(Output)

elif Example == 2:
    Input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    OR = np.array([0, 1, 1, 2])
    XOR = np.array([0, 1, 1, 0])
    AND = np.array([0, 0, 0, 1])

# Initialize network
NN = NeuralNetwork((np.size(Input[0]), 5, np.size(Output[0])))
# Train network
NN.train(Input, Output)

# Check
while True:
    feature_vector = raw_input('Introduce an input value (Format: input1, input2, ...): ')
    feature_vector = np.array(map(int, feature_vector.split(',')))
    print NN.forward_step(feature_vector)


