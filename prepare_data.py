import os
import gzip
import numpy as np
import torch
from sklearn.utils import shuffle  # For shuffling arrays in unison

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def prepare_data(path='mnist-data'):
    train_images, train_labels = load_mnist(path, kind='train')
    test_images, test_labels = load_mnist(path, kind='t10k')

    # Shuffle the training data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

    # Reshape the images to 28x28 pixels and normalize
    train_images = train_images.reshape(-1, 28, 28) / 255.0
    test_images = test_images.reshape(-1, 28, 28) / 255.0

    # Split the training data into training and validation sets
    tr_imgs, tr_labels = train_images[:50000], train_labels[:50000]
    val_imgs, val_labels = train_images[50000:], train_labels[50000:]

    # Convert to PyTorch tensors
    Xtr, Ytr = torch.tensor(tr_imgs, dtype=torch.float32), torch.tensor(tr_labels, dtype=torch.long)
    Xval, Yval = torch.tensor(val_imgs, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long)
    Xte, Yte = torch.tensor(test_images, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)

    return Xtr, Ytr, Xval, Yval, Xte, Yte
