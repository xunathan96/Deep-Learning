""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(1, dtype='f')
        
        self.Z = None
        self.truth = None
        self.batch_size = 0

    def forward(self, logit, gt):
        """
          Inputs: (minibatch)
          - logit: forward results from the last FCLayer, shape(batch_size, 10)
          - gt: the ground truth label, shape(batch_size, 10)
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.

        self.truth = gt
        self.batch_size = logit.shape[0]

        # Calculate Loss
        eS = np.exp(logit - np.max(logit, axis=1, keepdims=True))  # normalize logits for stability
        self.Z = eS / np.sum(eS, axis=1, keepdims=True)

        self.loss = -(1./self.batch_size)*np.sum(gt * np.log(self.Z))
        
        # Calculate Accuracy
        pred = np.argmax(self.Z, axis=1)
        truth = np.argmax(gt, axis=1)
        self.acc = np.mean(np.equal(pred, truth))
        
        ############################################################################
        
        return self.loss


    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)

        delta = (1./self.batch_size) * (self.Z - self.truth)
        return delta
    
        ############################################################################
