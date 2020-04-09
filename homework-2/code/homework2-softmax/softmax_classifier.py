import numpy as np
from numpy import linalg as LA
from scipy.special import logsumexp

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    
    # FORWARD PASS
    S = input @ W
    eS = np.exp(S)
    Z = eS / np.sum(eS, axis=1, keepdims=True)
    E = -np.sum(label * np.log(Z), axis=1)
    loss = np.mean(E) + 0.5*lamda*LA.norm(W)
    
    # PREDICTION
    prediction = np.argmax(Z, axis=1)
    
    # BACKPROPAGATION
    N = label.shape[0]
    gradient = (1./N) * input.T @ (Z - label) + lamda*W

    ############################################################################

    return loss, gradient, prediction
