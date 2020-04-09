""" Euclidean Loss Layer """

import numpy as np
    
class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.
        self.logit = None
        self.truth = None
        self.batch_size = 0.

    def forward(self, logit, gt):
        """
          Inputs: (minibatch)
          - logit: forward results from the last FCLayer, shape(batch_size, 10)
          - gt: the ground truth label, shape(batch_size, 10)
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.acc and self.loss respectively.
        # Only return the self.loss, self.acc will be used in solver.py.

        self.logit = logit
        self.truth = gt
        self.batch_size = logit.shape[0]
        
        # Calculate Loss
        l2Norm = np.sum((logit - gt)**2, axis=1)
        self.loss = 0.5*np.mean(l2Norm)
        
        # Calculate Accuracy
        pred = np.argmax(logit, axis=1)
        truth = np.argmax(gt, axis=1)
        self.acc = np.mean(np.equal(pred, truth))
        
        ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)

        delta = (1./self.batch_size) * (self.logit - self.truth)
        return delta

        ############################################################################
