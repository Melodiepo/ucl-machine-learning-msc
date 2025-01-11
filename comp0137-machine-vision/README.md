# COMP0137: Machine Vision

This repository contains courseworks, lecture materials, and lab practicals for COMP0137 Machine Vision. 

## Repository Structure

### `coursework-1`
This folder contains the materials for the first coursework assignment, focusing on Mixture of Gaussian and image processing tasks.

#### Contents:
- `machinevision-HW1.pdf`: Assignment brief for Coursework 1.
- `practicalMixGaussA.ipynb`, `practicalMixGaussB.ipynb`, `practicalMixGaussC.ipynb`, `practicalMixGaussD.ipynb`: Notebooks implementing Gaussian mixture models in different dimensions for image processing.
- `practicalMixGauss_Apples.ipynb`: A notebook applying Gaussian mixture models to identify apples in images.
- `apple_masks/` and `apple_photos/`: Supporting image datasets for the Gaussian mixture applications.

### `coursework-2`
This folder contains the materials for the second coursework assignment, focusing on camera models such as Homographies, Particle Filters, Condensations.

#### Contents:
- `HW2_Practical7c.ipynb`: Notebook implementing practical tasks for tracking and homography estimation.
- `HW2_TrackingAndHomographies.ipynb`: Notebook with implementations of tracking and homography techniques.
- `LLs.npy`, `LRs.npy`, `ULs.npy`, `URs.npy`: Numpy files containing supporting data for `HW2_TrackingAndHomographies.ipynb`.
- `Pattern01/`: Folder containing example patterns for analysis in `HW2_Practical7c.ipynb`.
- `labA.ipynb`, `labB.ipynb`: Lab notebooks for Condensation in Multi-camera models.
- `practical1A.ipynb`, `practical1B.ipynb`, `practical2A.ipynb`, `practical2B.ipynb`: Practical tasks in single camera model involving homographies.
- `ll.mat`, `lr.mat`, `ul.mat`, `ur.mat`: MATLAB files containing supporting datasets for `HW2_TrackingAndHomographies.ipynb`.

### `cs-labs`
This folder contains practical labs for different aspects of machine vision, organized by topics.

#### Subfolders:
- `01_FittingProbDistribs`: Probabilistic distribution fitting.
- `02_PracticalMixGauss`: Gaussian mixture models.
- `03_PracticalRegress_Pose`: Regression and pose estimation.
- `04_Classification`: Classification techniques in vision.
- `05_GraphicalModels`: Applications of graphical models.
- `06_Homographies`: Homography estimation and transformations.
- `07_Condensation`: Condensation algorithms for tracking.
- `08_NeuralNets`: Neural network-based vision models.
- `09_CNN`: Convolutional Neural Networks (CNNs).

### `lecture-notes-official`
This folder contains lecture slides provided for the module, covering a wide range of topics, including:
- Probability distributions
- Regression and classification
- Graphical models
- Homographies and pose estimation
- Neural networks and CNNs
- Temporal models and embeddings

The slides provide a comprehensive reference for the theoretical and practical aspects of machine vision.

### `lecture-notes-mine`
This folder contains personal notes summarizing the official lecture content:
- `comp0137-lec-notes-melodie-v1.pdf`: Detailed notes created for exam revision.

### Key Supporting Documents
- `MoGCribSheet.pdf`: A crib sheet summarizing Gaussian Mixture Model concepts.

WIP: Exam cheatsheet, some lab practices aren't fully done.