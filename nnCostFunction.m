function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

n = input_layer_size;
s2 = hidden_layer_size;
K = num_labels;
m = length(y);
Y = zeros(K,m);
Y(K*(0:m-1)+y')=1; %Y = K x m matrix

%Forward propagation
a1 = [ones(1,m) ; X']; %a1 = (n+1) x m matrix
z2 = Theta1*a1;
a2 = [ones(1,m) ; sigmoid(z2)]; %a2 = (s2+1) x m matrix
z3 = Theta2*a2;
a3 = sigmoid(z3); % a3 = K x m matrix

%Compute the cost function
J = -1/m * (Y(:)'*log(a3(:)) + (1-Y(:))'*log(1 - a3(:))) ...
    + lambda/(2*m) * (norm(reshape(Theta1(:,2:end),1,[]))^2 ...
                    + norm(reshape(Theta2(:,2:end),1,[]))^2);

%Backpropagation
d3 = a3 - Y;
d2 = (transpose(Theta2(:,2:end))*d3) .* sigmoid(z2) .* (1-sigmoid(z2));
Theta2_grad = 1/m * (d3*a2') + lambda/m * Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda/m * Theta2(:,1);
Theta1_grad = 1/m * (d2*a1') + lambda/m * Theta1;
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda/m * Theta1(:,1);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
