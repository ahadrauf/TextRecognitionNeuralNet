function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W is set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

% Notes: Partial inspiration for this function came from Andrew Ng's Coursera
%        machine learning course

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

EPSILON = 1e-4;
W = rand(L_out, L_in) * (2*EPSILON) - EPSILON;
W = [ones(L_out, 1) W];

% =========================================================================

end
