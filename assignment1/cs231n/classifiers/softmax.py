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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        f_i = X[i] @ W   #[1,c]
        f_i-=np.max(f_i)
        f_i_sum = np.sum(np.exp(f_i),keepdims=True)

        p = lambda k: np.exp(f_i[k]) / f_i_sum   #softmax 实现

        loss -= np.log(p(y[i]))

        # for j in range(num_class):
        #     if j==y[i]:
        #         dW[i][j]+=X[i]@(p(j)-1).T
        #     else:
        #         dW[i][j]+= X[i].reshape(1, -1).T @ f_i[i][j]

       # dW= X[i].reshape(1,-1).T@(p(y[i])-1)                      #[d,c]

        for k in range(num_class):
            p_k=p(k)
            dW[:,k] += (p_k-(k==y[i]))*X[i]                                   #[D,1]
            # print('dW',dW[:,k].shape)
            # print('X[i]',X[i].shape)
    loss /= num_train
    loss += 0.5*reg * np.sum(W ** 2)
    dW /= num_train
    dW += np.sum(W)

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
    num_class = W.shape[1]

    f = X@W #[n,c]
    f-=np.max(f,keepdims=True,axis=1)
    f_sum=np.sum(np.exp(f),axis=1,keepdims=True)
    p = np.exp(f) / f_sum
    loss+=np.sum(-np.log(p[np.arange(num_train),y]))


    tmp=np.zeros(p.shape)
    tmp[np.arange(num_train),y]=1
    dW=X.T@(p-tmp)



    loss /= num_train
    loss += 0.5 * np.sum(W ** 2)
    dW /= num_train
    dW += np.sum(W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
