import gzip
import struct
import numpy as np

MAX_BRIGHTNESS = 255

def read_training():
    return read_data('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz')


def read_testing():
    return read_data('./data/t10k-images-idx3-ubyte.gz', './data/t10k-labels-idx1-ubyte.gz')


def read_data(img_dir, label_dir):
    with gzip.open(img_dir) as image_file:
        struct.unpack('>4B', image_file.read(4))
        num_images = struct.unpack('>I', image_file.read(4))[0]
        num_pixels = struct.unpack('>I', image_file.read(4))[0] \
                     * struct.unpack('>I', image_file.read(4))[0]
        images = np.zeros((num_images, num_pixels))
        for i in range(num_images):
            images[i, :] = np.array(struct.unpack('>' + 'B' * num_pixels, image_file.read(num_pixels))) / MAX_BRIGHTNESS
    with gzip.open(label_dir) as labels_file:
        struct.unpack('>4B', labels_file.read(4))
        num_images = struct.unpack('>I', labels_file.read(4))[0]
        labels = np.zeros((num_images, 10))
        numerical_labels = []
        for i in range(num_images):
            label = np.zeros(10)
            numerical_label = struct.unpack('>B', labels_file.read(1))
            label[np.array(numerical_label)[0]] = 1.0
            labels[i, :] = label
            numerical_labels.append(numerical_label)
    return images, labels, numerical_labels
