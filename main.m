%% Inspiration for this project derived partially from Andrew Ng's
 % machine learning course on Coursera

%% Initialization
clear; close all; clc;

%% Setup fundamental parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Loading and Visualizing Data =============

% Load training data into X and y
fprintf('Loading and Visualizing Data ...\n')

load('data.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Loading Parameters ================

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('layerWeights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];


%% ================ Initializing Pameters ================
%  Create the initial neural net inputs by randomly initializing 
%  weights for the first and second layers of the neural net.
%  Stored the unrolled version into initial_nn_params to feed into
%  the fmincg function

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Training NN ===================

fprintf('\nTraining Neural Network... \n')

% 600 iterations worked well empirically
options = optimset('MaxIter', 600);

lambda = 1.48;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Visualize Weights =================
%  "Visualize" what the neural network is learning by displaying 
%  the hidden units to see what features they are capturing in 
%  the data.
%  This part is unnecessary (you can comment it out if unnecessary,
%  but it's good for debugging)

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Compute Network Accuracy =================

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%Theta1 = reshape(Theta1, ...
%                 hidden_layer_size, (input_layer_size + 1));

%Theta2 = reshape(Theta2, ...
%                 num_labels, (hidden_layer_size + 1));

%disp(Theta1);
%disp("\n");
%disp(Theta2);
