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
         
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% First computing the cost function
a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');

Y = zeros(size(a3));
for ii = 1:size(Y,1)
    Y(ii,y(ii))=1;
end
    
h_th = a3;
% Here is the cost function
J = (1/m)*sum(sum(-Y.*log(h_th)-(1-Y).*log(1-h_th)))+...
    (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


% Let's calculate the gradient

for ii=1:m
    % Forward propagation 
    a_1 = [1 X(ii,:)];
    
    z_2 = Theta1*a_1';
    a_2 = sigmoid(z_2);
    a_2 = [1;a_2];

    z_3 = Theta2*a_2;
    a_3 = sigmoid(z_3);
    
    % back-propagation 
    del_3 = a_3-(Y(ii,:))';
    del_2 = Theta2'*del_3.*sigmoidGradient([1; z_2]);
    
    Theta1_grad = Theta1_grad + del_2(2:end)*a_1;
    Theta2_grad = Theta2_grad + del_3*a_2';
end

Theta1_grad(:,1:1) = (1/m)*Theta1_grad(:,1:1);
Theta1_grad(:,2:end) = (1/m)*(Theta1_grad(:,2:end) + lambda*Theta1_grad(:,2:end));

Theta2_grad(:,1:1) = (1/m)*Theta2_grad(:,1:1);
Theta2_grad(:,2:end) = (1/m)*(Theta2_grad(:,2:end) + lambda*Theta2_grad(:,2:end));














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
