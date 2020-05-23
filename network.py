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
        for i in range(2, len(layers) + 1):
            self.weights[i] = np.random.randn(layers[i - 1], layers[i - 2])
            self.biases[i] = np.random.randn(layers[i - 1], 1)

    def classify(self, data):
        """
        Returns the classification of the provided data.
        """
        return np.argmax(self.feedforward(data))

    def feedforward(self, data):
        """
        Processes data represented as an n x 1 np array, where n is the size of the input layer.
        Returns an array of the activations in the last layer.
        """
        activations = data
        for i in range(2, self.L + 1):
            activations = sigmoid((self.weights[i] @ activations) + self.biases[i])
        return activations

    def feedforward_transparent(self, data):
        """
        Processes data represented as an n x 1 np array, where n is the size of the input layer.
        Returns the activations and z values of the neurons in each layer.
        """
        activations = {1: data}
        z_values = {}
        for i in range(2, self.L + 1):
            z = self.weights[i] @ activations[i - 1] + self.biases[i]
            z_values[i] = z
            activations[i] = sigmoid(z)
        return activations, z_values

    def backpropagate(self, data, label):
        """
        Runs backpropagation with the given data and label. Returns the weight gradient and bias gradient.
        """
        activations, z_values = self.feedforward_transparent(data)
        errors = {self.L: (activations[self.L] - label) * sigmoid_derivative(z_values[self.L])}
        wg = {self.L: errors[self.L] @ activations[self.L - 1].T}
        bg = {self.L: errors[self.L]}
        for layer in range(self.L - 1, 1, -1):
            errors[layer] = ((self.weights[layer + 1]).T @ errors[layer + 1]) * sigmoid_derivative(z_values[layer])
            wg[layer] = errors[layer] @ activations[layer - 1].T
            bg[layer] = errors[layer]
        return wg, bg

    def epoch(self, data, labels, batch_size, eta):
        """
        Trains the network for 1 epoch, given a dataset, corresponding labels, a desired batch size, and desired
        step size eta.
        """
        start = 0
        while start < np.shape(data)[0]:
            curr_size = min(len(data) - start, batch_size)
            wg = {j:np.zeros(np.shape(self.weights[j])) for j in range(2, self.L + 1)}
            bg = {j:np.zeros(np.shape(self.biases[j])) for j in range(2, self.L + 1)}
            for point in range(start, start + curr_size):
                wg_addend, bg_addend = self.backpropogate(data[[point], :].T, labels[[point], :].T)
                for layer in range(2, self.L + 1):
                    wg[layer] += wg_addend[layer]
                    bg[layer] += bg_addend[layer]
            self.gradient_descent(wg, bg, curr_size, eta)
            start += curr_size

    def gradient_descent(self, wg, bg, m, eta):
        for layer in range(2, self.L + 1):
            self.weights[layer] -= (eta / m) * wg[layer]
            self.biases[layer] -= (eta / m) * bg[layer]


def sigmoid(x):
    """
    Applies the sigmoid function element-wise to an ndarray, x.
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Finds the element-wise sigmoid derivative for an ndarray, x.
    :param x:
    :return:
    """
    return sigmoid(x) * (1.0 - sigmoid(x))
