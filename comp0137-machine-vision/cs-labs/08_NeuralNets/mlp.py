# Multilayer Perceptron
# Author: Daniel E. Worrall, 3 Dec 2016

import sys
sys.path.append('layers')

import relu_layer
import affine_layer
import crossentropy_softmax_layer
import softmax_layer

import numpy as np

from matplotlib import pyplot as plt

class Mlp(object):

    def __init__(self):
        self.net = [] # the network

    def run(self, step_number=10000, disp=True):
        # The structure of the code below is designed to 
        # highlight the modular, reusable nature of neural 
        # network layers. Each layer is defined in a file 
        # with name "<name>_layer.py", for example, the ReLU 
        # layer is in relu_layer.m. Each layer file has at 
        # least 2 'methods'. There is a 'forward' method, 
        # which implements the  forward-propagation step 
        # needed to compute activations, and a 'backward' 
        # method, which implements the back-propagation step. 

        # Generate data
        X, t = generate_data()

        # Find the limits of the data and add boundary for plotting
        min_ = np.amin(X, axis=0)
        max_ = np.amax(X, axis=0)

        diff = max_ - min_
        min_ = min_ - diff/3
        max_ = max_ + diff/3

        # Create a dense grid of testpoints (test_coords), which we shall use to
        # visualize the forward model p(y|x)
        I, J = np.meshgrid(np.linspace(min_[0], max_[0]), np.linspace(min_[1], max_[1]))
        test_coordinates = np.reshape(np.stack([I,J], axis=-1), [-1,2])

        ## Build MLP
        # Parameters for stochastic gradient descent
        minibatch_size = 10
        initial_learning_rate = 1e-2

        # Construct the network as an ordered cell array, where each element is a
        # layer
        self.net = build_mlp(X.shape[1], 250, t.shape[1])

        if disp:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            plt.ion()

            fig.show()
            fig.canvas.draw()
        ## Training loop

        loss_layer = crossentropy_softmax_layer.Crossentropy_softmax_layer()

        for i in range(step_number):
            # Adaptive learning rate satisfying Robbins-Monro conditions
            learning_rate = initial_learning_rate/np.sqrt(i+1)

            # Minibatching
            mb = np.random.randint(0, high=200, size=minibatch_size)
            xmb = X[mb,:]
            tmb = t[mb,:]

            # Complete the forward propagation step
            logits, self.net = mlp_forward(self.net, xmb)

            # We construct a separate loss layer on the end of the network, which
            # we interchange for the softmax at test-time. By merging the
            # crossentropy and softmax layers at training time, we avoid numerical
            # precision problems during gradient descent.

            # Complete the forward propagration step
            loss = loss_layer.forward(logits, tmb)

            # Implement the backward pass
            dLdy = loss_layer.backward(1.)
            self.net = mlp_backward(self.net, dLdy)

            # Implement stochastic gradient descent code
            self.net = apply_gradient_descent_step(self.net, learning_rate)

            # Validation
            # Get training accuracy on this minibatch
            indy = np.argmax(logits, axis=1)
            indt = np.argmax(tmb, axis=1)
            accuracy = np.mean(indt==indy)

            # Test
            if i % 10 == 0:
                sys.stdout.write("[{:04d}], Loss: {:04f}, Accuracy: {:04f}, Learning rate: {:04f}\r".format(i, loss, accuracy, learning_rate))
                sys.stdout.flush()
                # Run test data through mlp
                test_logits, __ = mlp_forward(self.net, test_coordinates)
                test_output = softmax_layer.Softmax_layer().forward(test_logits)
                
                if disp:
                    # Plot mesh
                    ax.clear()

                    ax.pcolor(I,J,np.reshape(test_output[:,0], I.shape))
                    ax.contour(I,J,np.reshape(test_output[:,0], I.shape), origin='lower', levels=[0.5,], colors='r')
                    ax.scatter(X[:,0], X[:,1], 10, np.concatenate([t, np.ones((200,1))], axis=1))
                    fig.canvas.draw()


def generate_data():
    mean1 = np.asarray([0.,0.])[np.newaxis,:]
    std1 = 0.2
    mean2 = np.asarray([0.,1.])[np.newaxis,:]
    std2 = 0.2
    # We shall generate two interlocking arcs
    theta1 = 1.2*np.pi*(np.random.rand(100,1)-0.5)
    X1 = mean1 + np.concatenate([np.cos(theta1),np.sin(theta1)], axis=1) + std1*np.random.randn(100,2)
    theta2 = 1.2*np.pi*(np.random.rand(100,1)+0.5)
    X2 = mean2 + np.concatenate([np.cos(theta2),np.sin(theta2)], axis=1) + std2*np.random.randn(100,2)

    # Input X and target t
    X = np.concatenate([X1, X2], axis=0)
    t0 = np.concatenate([np.ones((100,1)), np.zeros((100,1))], axis=0)
    t1 = np.concatenate([np.zeros((100,1)), np.ones((100,1))], axis=0)
    t = np.concatenate([t0, t1], axis=1)
    return X, t


## Functions
def build_mlp(n_in, n_hid, n_out):
    # Construct a neural network as a list. Each element of the
    # list is a layer. Just make sure that the dimensionality of 
    # each layer is consistent with its neighbors.
    
    # Declare each layer
    affine1 = affine_layer.Affine_layer(n_in, n_hid)
    relu1 = relu_layer.ReLU_layer()   # ReLU doesn't alter dimensions -> no dim. args
    affine2 = affine_layer.Affine_layer(n_hid, n_out)

    # Build network as ordered cell array
    network = [affine1, relu1, affine2]
    return network


def mlp_forward(net, x):
    # Each layer takes as input the output y from the layer below.

    y = x
    for layer in net:
        y = layer.forward(y)
    return y, net


def mlp_backward(net, dLdy):
    # Back-propagation: Each layer takes as input the back-propagated 
    # errors/gradients/deltas df from the layer above.
    #
    # CAVEAT: Gradient computation relies not only on dLdy but also x and y 
    # from forward-propagation step, these are stored in-place in each layer, 
    # so you can only run this method after calling 
    # mlp_forward(net, x, train=true).

    # Implement the backward pass
    for layer in reversed(net):
        dLdy = layer.backward(dLdy)
    return net


def apply_gradient_descent_step(net, learning_rate):
    # Gradient descent step: Apply simple stochastic gradient descent step
    for layer in net:
        if hasattr(layer, 'dLdW'):
            # TODO: update the weights of the multilayer perceptron
            layer.W = layer.W - learning_rate*layer.dLdW

    return net



