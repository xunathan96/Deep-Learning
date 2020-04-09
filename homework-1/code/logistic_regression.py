import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_data_loader

def loadData():
    mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
    # training dataset
    train_set = mnist_dataset.train
    # test dataset
    test_set = mnist_dataset.test
    print("Training dataset size: " , train_set.num_examples)
    print("Test dataset size: ", test_set.num_examples)
    return train_set, test_set

train_data, test_data = loadData()
# convert labels to 0 or 1 to allow us to work with logistic regression
train_data.labels[train_data.labels == 3] = 0
train_data.labels[train_data.labels == 6] = 1
test_data.labels[test_data.labels == 3] = 0
test_data.labels[test_data.labels == 6] = 1

# append 1 to features to integrate bias term
bias = np.ones(train_data.images.shape[0])
train_data.images = np.c_[train_data.images, bias]
bias = np.ones(test_data.images.shape[0])
test_data.images = np.c_[test_data.images, bias]

def binary_cross_entropy(z, t):
    z[z==0] = 1e-25  # prevent divide by 0 error
    bce = -np.mean(t*np.log(z) + (1-t)*np.log(1-z))
    return bce

def forward_pass(X, w, t):
    s = X @ w
    z = 1./(1. + np.exp(-s))
    return z, s

def backward_pass(X, z, t):
    grad_w = np.mean(X * (z-t)[:,None], axis=0)
    return grad_w

def get_accuracy(z, t):
    return np.mean(np.equal((z>0.5), t))

def train_model():

    # initialize w as vector of dimension 28*28 + 1 == 784 + 1
    w = np.random.normal(size=train_data.images.shape[1])

    loss = {'train': [], 'test': []}
    accuracy = {'train': [], 'test': []}
    learning_rate = 0.01
    batch_size = 32
    max_epoch = 50

    for epoch in range(0, max_epoch):
        iter_per_batch = train_data.num_examples // batch_size
        for batch_id in range(0, iter_per_batch):
            # GET MINI-BATCH DATA
            batch = train_data.next_batch(batch_size)
            input, label = batch

            # COMPUTE PREDICTIONS
            z, s = forward_pass(input, w, label)

            # CALCULATE LOSS & ACCURACY
            bce = binary_cross_entropy(z, label)
            acc = get_accuracy(z, label)

            # UPDATE WEIGHTS
            grad_w = backward_pass(input, z, label)
            w = w - learning_rate * grad_w


        # RECORD TRAINING PERFORMANCE
        loss['train'].append(bce)
        accuracy['train'].append(acc)

        # RECORD TEST PERFORMANCE
        z_test, _ = forward_pass(test_data.images, w, test_data.labels)
        bce_test = binary_cross_entropy(z_test, test_data.labels)
        acc_test = get_accuracy(z_test, test_data.labels)
        loss['test'].append(bce_test)
        accuracy['test'].append(acc_test)

    return loss, accuracy


def plot_curves(loss, accuracy):

    # PLOT LOSS CURVES
    plt.plot(loss['train'], color='blue', label='training data')
    plt.plot(loss['test'], color='green', label='test data')
    plt.legend()
    plt.title('Loss')
    plt.ylabel('Binary Cross Entropy')
    plt.xlabel('Epoch')
    plt.show()

    # PLOT ACCURACY CURVES
    plt.plot(accuracy['train'], color='blue', label='training data')
    plt.plot(accuracy['test'], color='green', label='test data')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    loss, accuracy = train_model()
    plot_curves(loss, accuracy)
