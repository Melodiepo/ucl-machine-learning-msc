"""Assignment 1, Unsupervised Learning, UCL 2003
Author: Zoubin Gahramani
Ported to Python by Raza Habib and Jamie Townsend 2017"""
import numpy as np
from matplotlib import pyplot as plt

# Python comments use a hash

def main():
    # load the data set
    Y = np.loadtxt('binarydigits.txt')
    N, D = Y.shape

    # this is how you display one image using matplotlib,
    # e.g. the 4th image:
    y = Y[3,  :]
    plt.figure()
    plt.imshow(np.reshape(y, (8,8)),
               interpolation="None",
               cmap='gray')
    plt.axis('off')

    # now we will display the whole data set:
    plt.figure(figsize=(5, 5))
    for n in range(N):
        plt.subplot(10, 10, n+1)
        plt.imshow(np.reshape(Y[n, :], (8,8)),
                   interpolation="None",
                   cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
