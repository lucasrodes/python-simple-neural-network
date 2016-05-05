import numpy


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
                                self.neurons_per_layer[layer_index])  # Create new neuron layer
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

    def init_weights(self):
        b = 0

    def forward(self, x):
        a = 0

    def train(self):
        a = 0

class NeuronLayer:

    # Initialize the neuron layer
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weight_matrix = self.init_weights

    # Initialize the weights of this layer, i.e. the weight matrix W
    def init_weights(self):
        # Mean and standard deviation of the initialization of the weight matrix
        mu = 0
        sigma = 0.5
        self.weight_matrix = numpy.random.normal(mu, sigma, [self.num_neurons, self.num_inputs])


a = NeuralNetwork((2, 3, 2))
a.check()