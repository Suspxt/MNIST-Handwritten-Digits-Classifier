import numpy as np
import data_reader
from network import Network


training_data, training_labels, _ = data_reader.read_training()
testing_data, testing_labels, testing_numerical_labels = data_reader.read_testing()
num_tests = np.shape(testing_data)[0]
layers = input("Specify the desired layers of the network, separated by spaces. Enter nothing for [784, 16, 16, 10].")

if not layers:
    layers = [784, 16, 16, 10]
else:
    layers = [int(layer) for layer in layers.split()]
network = Network(layers)

epochs = int(input("Enter the number of epochs to train for. Enter nothing for 30 epochs.") or 30)
batch_size = int(input("Enter the desired batch size. Enter nothing for a batch size of 10.") or 10)
eta = int(input("Enter the desired step size for gradient descent. Enter nothing for a step size of 3.") or 3)

num_correct = 0
for i in range(num_tests):
    if network.classify(testing_data[i, :]) == testing_numerical_labels[i]:
        num_correct += 1
print("Epoch 0: {} / {}".format(num_correct, num_tests))

for epoch in range(1, epochs + 1):
    network.epoch(training_data, training_labels, batch_size, eta)
    num_correct = 0
    for i in range(num_tests):
        if network.classify(testing_data[i, :]) == testing_numerical_labels[i]:
            num_correct += 1
    print("Epoch {}: {} / {}".format(epoch, num_correct, num_tests))
