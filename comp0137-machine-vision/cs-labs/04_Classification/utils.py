import matplotlib.pyplot as plt
import numpy as np


def add_bias(arr, axis=0):
    """Append ones to the given axis so that a bias can be used with matrix multiplication.

    Args:
        arr (ndarray): input array of arbitrary dimensions
        axis (int, optional): axis to add bias to.

    Returns:
        ndarray: copy of array where ones have been appended on the given axis

    """
    bias_shape = list(arr.shape)
    bias_shape[axis] = 1
    bias = np.ones(bias_shape)
    return np.concatenate((bias, arr), axis)


def visualise_faces(face_data, grid=(4, 4)):
    """Visualise labeled face images from the given dataset.

    Args:
        face_data (tuple): pair of iterables that give 24x24 greyscale images and their class label respectively.
        grid (tuple): dimensions of the grid of images

    """
    fig, axes = plt.subplots(*grid, figsize=(9, 8), squeeze=True)
    for ax, image, is_face in zip(axes.flatten(), *face_data):
        ax.imshow(image.reshape(24, 24), cmap='gray')
        ax.axis('off')
        ax.set(title=('Face' if is_face else 'Non-Face'))
    plt.tight_layout()
    plt.show()
