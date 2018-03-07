# TextRecognitionNeuralNet
This set of MATLAB functions implements a two-layer neural net to identify hand-written text to up to 99% accuracy.

The `main.m` function runs through the entire training and and testing process for a preloaded set of data (contained in `data.mat`). To modify the data input, you can set this file in the `main.m` section labeled "Loading and Visualizing Data". The minimization function used is Carl Edward Rasmussen's 2001-2002 fmincg, which is well-suited for fast and accurate minimization of low-dimensional data sets. Classification is done using a multiple-output sigmoid classifier (implemented in `sigmoidGradient.m`).

I credit a large portion of the training data and some of the image visualization code in `main.m` to Andrew Ng's online Coursera course's homework #4 (found at https://www.coursera.org/learn/machine-learning). 
