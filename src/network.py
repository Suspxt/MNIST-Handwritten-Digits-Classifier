import numpy as np

class Network:
    def __init__(self, layers):
        """
        Layers is a list of the number of neurons in each layer, with layers[0] being the number of
        neurons in the input layer.
        """
        self.weights = []
        self.biases = []
        self.layers = layers
        for i in range(len(layers) - 1):
            self.weights[i] = np.random.randn(layers[i + 1], layers[i])
            self.biases[i] = np.random.randn(layers[i + 1], 1)

    def feedforward(self, data):
        """
        Classifies data represented as an n x 1 np array, where n is the size of the input layer.
        Returns an array of the activations in the last layer.
        """
        activations = data
        for i in range(len(self.weights) - 1):
            activations = relu(self.weights[i] @ activations + self.biases[i])
        return activations

    def feedforward_transparent(self, data):
        activations = [np.zeros((self.layers[size], 1)) for size in self.layers]
        activations[0, :] = data
        z_values = []
        for i in range(len(self.weights) - 1):
            z = self.weights[i + 1] @ activations[i] + self.biases[i + 1]
            z_values.append(z)
            activations[i, :] = relu(z)
        return activations, z_values

    def backpropogate(self, data, label):
        activations, z_values = self.feedforward_transparent(data)
        wg = [np.zeros((self.layers[j + 1], self.layers[j])) for j in range(len(self.layers) - 1)]
        bg = [np.zeros((self.layers[j], 1)) for j in range(1, len(self.layers))]
        for j in range(len(wg[-1])):
            for k in range(len(wg[-2])):
                partial_wg = activations[-2, k] * relu_derivative(z_values[-1]) * (activations[-1] - label)
        dC_dActivations = activations[-1] - label
        for layer in range(len(wg) - 2, -1, -1):
            weights = self.weights[layer]
            for j in range(weights.shape[0]):
                for k in range(weights.shape[1]):
                    #todo
                    partial_wg = activations[layer - 1, k] * relu_derivative(z_values[layer]) * dC_dActivations #todo finish this eq


    def train(self, data, labels, batch_size, eta):
        i = 0
        while i < len(data):
            curr_size = min(len(data) - i, batch_size)
            weight_gradient = [np.zeros((self.layers[j + 1], self.layers[j])) for j in range(len(self.layers) - 1)]
            bias_gradient = [np.zeros((self.layers[j], 1)) for j in range(1, len(self.layers))]
            for j in range(i, i + curr_size):
                wg_addend = self.backpropogate(data[j, :], labels[j, :])

            i += curr_size

    def evaluate(self, data, labels):
        pass


def relu(self, x):
    if x >= 0:
        return x
    else:
        return 0


def relu_derivative(self, x):
    if x >= 0:
        return 1
    else:
        return 0


