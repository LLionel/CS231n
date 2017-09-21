import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #print('dW.shape=',dW.shape)
        #print('X[i].shape=',X[i].shape)
        dW[:,j] += X[i]
        dW[:,y[i]] += -X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = np.reshape(scores[range(num_train),y],(num_train,1))
  margins = scores-correct_class_scores+1
  margins[range(num_train),y] = 0;
  margins[margins<0] = 0
  loss = np.sum(margins)/num_train
  loss += reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  nonzero_scores_x , nonzero_scores_y = np.nonzero(margins);
  """
  #Debug information
  dW[:,nonzero_scores_y] += X[nonzero_scores_x,:].T
  A = np.array([[1,2,3],[4,5,6],[7,8,9]])
  B = np.zeros((2,3));
  C = np.array([[1,1,1],[2,2,2]])
  print(A)
  B[0] = A[0]
  B[1] = A[0]
  print('B=')
  print(B)
  B += C
  print('after changing...')
  print(B)
  print('after changing...')
  print(A)
  #print(dW)
  #print(np.linalg.norm(dW - old, ord='fro'))
  #dW[:,y[nonzero_scores_x]] -= np.sum(margins[nonzero_scores_x,:],axis=1)*X[nonzero_scores_x,:].T
  dW[:,y[nonzero_scores_x]] += -X[nonzero_scores_x,:].T
  dW /= num_train
  dW += 2*reg*W
  """
  
  margins[margins>0] = 1
  row_sum = np.sum(margins,axis=1)
  margins[np.arange(num_train),y] = -row_sum
  dW += np.dot(X.T,margins)/num_train +2*reg*W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
