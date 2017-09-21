import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    #origin_score.shape = (10,)
    origin_score = X[i].dot(W)
    #exp_score.shape = (10,)
    exp_score = np.exp(origin_score)
    sum_score = np.sum(exp_score)
    nom_score = exp_score/sum_score
    #score = -np.log(nom_score)
    loss -= np.log(nom_score[y[i]])
    
    dW += np.reshape(X[i],(X[i].size,1)).dot(np.reshape(exp_score,(1,exp_score.size)))/sum_score
    dW[:,y[i]] += -X[i] #+ X[i]*exp_score[y[i]]/sum_score
  loss = loss/num_train + reg*np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  origin_score = X.dot(W)
  origin_score = origin_score - np.max(origin_score,axis=1,keepdims=True)
  exp_score = np.exp(origin_score)
  sum_score = np.sum(exp_score,axis=1,keepdims=True)
  #print('sum_score.shape=',sum_score.shape)
  nom_score = exp_score/sum_score
  loss += -np.sum(np.log(nom_score[range(num_train),y]))/num_train + reg*np.sum(W*W)
  '''
  dW = X.T.dot(exp_score)
  temp_W = np.zeros_like(W)
  temp_W[:,y] = 1
  dW -= X.T.dot(temp_W)
  '''
  dW = X.T.dot(exp_score/sum_score)
  temp = np.zeros_like(origin_score)
  temp[range(num_train),y] = 1
  dW -= X.T.dot(temp) 
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

