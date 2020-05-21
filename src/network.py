import numpy as np

class Network:
    def __init__(self, layers):
        """
        Layers is a list of the number of neurons in each layer, with layers[0] being the number of
        neurons in the input layer.
        """
        self.weights = {}
        self.biases = {}
        self.layers = layers  # length L
        self.L = len(layers)
        for i in range(2, len(layers)):
            self.weights[i] = np.random.randn(layers[i - 1], layers[i - 2])
            self.biases[i] = np.random.randn(layers[i - 1], 1)

    def feedforward(self, data):
        """
        Classifies data represented as an n x 1 np array, where n is the size of the input layer.
        Returns an array of the activations in the last layer.
        """
        activations = data
        for i in range(2, self.L):
            activations = relu(self.weights[i] @ activations + self.biases[i])
        return activations

    def feedforward_transparent(self, data):
        activations = {1: data}
        z_values = {}
        for i in range(2, self.L):
            z = self.weights[i] @ activations[i - 1] + self.biases[i]
            z_values[i] = z
            activations[i] = relu(z)
        return activations, z_values

    def backpropogate(self, data, label):
        activations, z_values = self.feedforward_transparent(data)
        errors = {self.L: (activations[-1] - label) * relu_derivative(z_values[-1])}
        wg = {self.L: errors[self.L] @ activations[self.L - 1].T}
        bg = {self.L: errors[self.L]}
        for layer in range(self.L - 1, 1, -1): # L - 1 to 2
            errors[layer] = ((self.weights[layer + 1]).T @ errors[layer + 1]) * relu_derivative(z_values[layer])
            wg[layer] = errors[layer] @ activations[layer - 1].T
            bg[layer] = errors[layer]
        return wg, bg


    def train(self, data, labels, batch_size, eta):
        i = 0
        while i < len(data):
            curr_size = min(len(data) - i, batch_size)
            wg = {(j, np.zeros((self.layers[j + 1], self.layers[j]))) for j in range(self.L - 1)}
            bg = {np.zeros((self.layers[j], 1)) for j in range(1, self.L)}
            for j in range(i, i + curr_size):
                wg_addend, bg_addend = self.backpropogate(data[j, :], labels[j, :])

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


