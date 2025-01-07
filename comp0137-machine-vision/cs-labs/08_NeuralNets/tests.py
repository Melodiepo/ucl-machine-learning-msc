# Practical 10a tests
# Author: Daniel E. Worrall, 11 Dec 2017

# Fix random seed
import sys
sys.path.append('layers')

import numpy as np

import relu_layer
import affine_layer
import crossentropy_softmax_layer


def relu_forward_test():
    np.random.seed(seed=1337)
    relu = relu_layer.ReLU_layer()

    test_x = np.asarray([[-0.1, 0.1], [0.1, -0.1]])
    test_y = np.asarray([[0.0, 0.1], [0.1, 0.0]])

    y = relu.forward(test_x)

    if (y.size != test_y.size):
        sys.exit('TODO 1.1 ReLU forward pass failed: Output wrong size')
    elif np.any(np.abs(y-test_y)>1.0e-6):
        sys.exit('TODO 1.1 ReLU forward pass failed: Output value incorrect')
    else:
        print('ReLU forward pass test correct.')


def relu_backward_test():
    np.random.seed(seed=1337)
    relu = relu_layer.ReLU_layer()

    test_x = np.asarray([[-0.1, 0.1], [0.1, -0.1]])
    test_dldy = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    test_dldx = np.asarray([[0.0, 0.2], [0.3, 0.0]])

    y = relu.forward(test_x)
    dldx = relu.backward(test_dldy)

    if (dldx.size != test_dldx.size):
        sys.exit('TODO 1.2 ReLU backward pass failed: Update wrong size')
    elif np.any(np.abs(dldx-test_dldx)>1.0e-6):
        sys.exit('TODO 1.2 ReLU backward pass failed: Update wrong value')
    else:
        print('ReLU backward pass test correct.')


def affine_forward_test():
    np.random.seed(seed=1337)
    affine = affine_layer.Affine_layer(3, 2)

    test_x = np.asarray([[-0.1, 0.2, 0.3], [0.1, -0.2, -0.3]])
    test_y = np.asarray([[0.04923396, -0.51271381], [-0.02923396, 0.53271381]])

    y = affine.forward(test_x)

    if (y.size != test_y.size):
        sys.exit('TODO 2.1 Affine forward pass failed: Output wrong size')
    elif np.any(abs(y-test_y)>1.0e-6):
        sys.exit('TODO 2.1 Affine forward pass failed: Output value incorrect')
    else:
        print('Affine forward pass test correct.')


def affine_backward_test():
    np.random.seed(seed=1337)
    affine = affine_layer.Affine_layer(3, 2)

    test_x = np.asarray([[-0.1, 0.2, 0.3], [0.1, -0.2, -0.3]])
    test_y = np.asarray([[0.04923396, -0.51271381], [-0.02923396, 0.53271381]])
    test_dldy = np.asarray([[-0.1, -0.2], [0.3, 0.4]])
    test_dldx = np.asarray([[0.09721147, 0.22123899, 0.22030905], [-0.23502148, -0.46105794, -0.42868632]])

    y = affine.forward(test_x)
    dldx = affine.backward(test_dldy)

    if (y.size != test_y.size):
        sys.exit('TODO 2.2 Affine backward pass failed: Update wrong size')
    elif np.any(abs(dldx-test_dldx)>1.0e-6):
        sys.exit('TODO 2.2 Affine backward pass failed: Update value incorrect')
    else:
        print('Affine backward pass test correct.')


def crossentropy_softmax_softmax_test():
    np.random.seed(seed=1337)
    crossentropy_softmax = crossentropy_softmax_layer.Crossentropy_softmax_layer()

    test_x = np.asarray([[-0.1, 0.3], [0.2, -0.3]])
    test_target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    test_softmax_output = np.asarray([[0.401312339887548, 0.598687660112452], [0.622459331201855, 0.377540668798146]])

    y = crossentropy_softmax.forward(test_x, test_target)
    softmax_output = crossentropy_softmax.softmax_output

    if (softmax_output.size != test_softmax_output.size):
        sys.exit("TODO 3.1 Cross-entropy-softmax softmax failed: Output wrong size")
    elif np.any(np.abs(softmax_output-test_softmax_output)>1.0e-6):
        sys.exit("TODO 3.1 Cross-entropy-softmax softmax failed: Output value wrong")
    else:
        print("Cross-entropy softmax layer's softmax output test correct.")


def crossentropy_softmax_forward_test():
    np.random.seed(seed=1337)
    crossentropy_softmax = crossentropy_softmax_layer.Crossentropy_softmax_layer()

    test_x = np.asarray([[-0.1, 0.3], [0.2, -0.3]])
    test_target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    test_y = np.asarray([0.49354611])

    y = crossentropy_softmax.forward(test_x, test_target)

    if (y.size != test_y.size):
        sys.exit('TODO 3.2 Cross-entropy-softmax forward pass failed: Output wrong size')
    elif np.any(np.abs(y-test_y)>1.0e-6):
        sys.exit('TODO 3.2 Cross-entropy-softmax forward pass failed: Output value incorrect')
    else:
        print('Cross-entropy softmax forward pass test correct.')


def crossentropy_softmax_backward_test():
    np.random.seed(seed=1337)
    crossentropy_softmax = crossentropy_softmax_layer.Crossentropy_softmax_layer()

    test_x = np.asarray([[-0.1, 0.3], [0.2, -0.3]])
    test_target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    test_dldy = -0.1
    test_dldx = np.asarray([[0.401312339887548, -0.401312339887548], [-0.377540668798145, 0.377540668798146]])

    y = crossentropy_softmax.forward(test_x, test_target)
    dldx = crossentropy_softmax.backward(test_dldy)

    if (dldx.size != test_dldx.size):
        sys.exit('TODO 3.3 Cross-entropy softmax backward pass failed: Update wrong size')
    elif np.any(np.abs(dldx-test_dldx)>1.0e-6):
        sys.exit('TODO 3.3 Cross-entropy softmax backward pass failed: Update value incorrect')
    else:
        print('Cross-entropy softmax backward pass test correct.')


def apply_gradient_descent_step_test():
    """Check if the backprop step works"""
    np.random.seed(seed=1337)
    test_net = np.load('test_net.npz')
    FLAG = False
    
    import mlp
    nn = mlp.Mlp()
    minibatch_size = 10
    learning_rate = 1e-2
    
    X, t = mlp.generate_data()
    my_net = mlp.build_mlp(X.shape[1], 250, t.shape[1])

    loss_layer = crossentropy_softmax_layer.Crossentropy_softmax_layer()

    # Minibatching
    mb = np.random.randint(0, high=200, size=minibatch_size)
    xmb = X[mb,:]
    tmb = t[mb,:]

    # Complete the forward propagation step
    logits, my_net = mlp.mlp_forward(my_net, xmb)

    # Complete the forward propagration step
    loss = loss_layer.forward(logits, tmb)

    # Implement the backward pass
    dLdx = loss_layer.backward(1.)
    my_net = mlp.mlp_backward(my_net, dLdx)

    # Implement stochastic gradient descent code
    my_net = mlp.apply_gradient_descent_step(my_net, learning_rate)

    shapes = [(3, 250), (251, 2)]
    layers = ['affine1', 'affine2']
    for layer in my_net:
        if hasattr(layer, 'W'):
            W_shape = layer.W.shape
            if W_shape not in shapes:
                print()
                sys.exit('TODO 4. Backprop test failed: Update wrong shape')
            else:
                layer_weights = test_net[layers[shapes.index(W_shape)]]
                if np.any(np.abs(layer_weights - layer.W) >= 1e-6):
                    sys.exit('TODO 4. Backprop test failed: Update value incorrect')
                
    print()
    print('Backprop pass test passed.')

