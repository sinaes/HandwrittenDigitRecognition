function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

p = zeros(size(X, 1), 1);

% setting our training dataset as the first layer's units also adding a
% bias value to it
a_1 = [ones(m,1) X];

% Computing the the second layer's units
a_2 = sigmoid(a_1*Theta1');
a_2 = [ones(m,1) a_2];

% Computing the last layer's units this is the final layer which gives
% us the probability
h_th = sigmoid(a_2*Theta2');

% "p" is our prediction
[~,p]=max(h_th,[],2);








% =========================================================================


end
