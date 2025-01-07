import sys

import numpy as np

sys.path.append('layers')
import relu_layer
import affine_layer
import sigmoid_layer

import matplotlib.pyplot as plt

def build_encoder(weights):
    # Construct a neural network as an ordered cell array. Each element of the
    # cell array is a layer. Just make sure that the dimensionality of each
    # layer is consistent with its neighbors.

    # Declare each layer
    affine_e1 = affine_layer.Affine_layer(784, 500)
    relu_e1 = relu_layer.ReLU_layer()  
    affine_e2 = affine_layer.Affine_layer(500, 500)
    relu_e2 = relu_layer.ReLU_layer()
    mu_e = affine_layer.Affine_layer(500,500)

    # Load pretrained weights
    affine_e1.W = weights['encoder_W_e1']
    affine_e2.W = weights['encoder_W_e2']
    mu_e.W = weights['encoder_W_e_mu']

    # Build network as ordered cell array
    return [affine_e1, relu_e1, affine_e2, relu_e2, mu_e]


def build_decoder(weights):
    affine_d1 = affine_layer.Affine_layer(500, 500)
    relu_d1 = relu_layer.ReLU_layer()
    affine_d2 = affine_layer.Affine_layer(500, 500)
    relu_d2 = relu_layer.ReLU_layer()
    affine_d3 = affine_layer.Affine_layer(500,784)
    sigmoid_d = sigmoid_layer.Sigmoid_layer()

    affine_d1.W = weights['binary_decoder_W_d1']
    affine_d2.W = weights['binary_decoder_W_d2']
    affine_d3.W = weights['binary_decoder_W_d_mu']
    return [affine_d1, relu_d1, affine_d2, relu_d2, affine_d3, sigmoid_d]


def plot_tiled_array(images, title):
    # Draw images in tiled-array
    n_images = images.shape[0]
    length = int(np.sqrt(images.shape[1]))
    sqrt_nimages = int(np.ceil(np.sqrt(n_images)))
    tiled_image = np.zeros((length*sqrt_nimages,length*sqrt_nimages))

    for i in range(n_images):
        tile = np.reshape(images[i,:], [length,length])
        m = int(length*np.floor((1.*i)/sqrt_nimages))
        n = int(np.mod(i,sqrt_nimages)*length)
        tiled_image[m:(m+length),n:(n+length)] = tile

    plt.imshow(1-tiled_image, cmap="gray")
    plt.title(title)
    plt.show()
