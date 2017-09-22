function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

% hypothesis function
h_th = sigmoid(X*theta);

% Our cost function with regularized factor(lambda)
J = (1/m)*sum(-y.*log(h_th)-(1-y).*log(1-h_th))+(lambda/(2*m))*sum(theta(2:end).^2);

% The fradient of cost function
grad(1) = (1/m)*(X(:,1))'*(h_th-y);
grad(2:end) = (1/m)*(X(:,2:end))'*(h_th-y) + (lambda/m)*theta(2:end);







% =============================================================

grad = grad(:);

end
