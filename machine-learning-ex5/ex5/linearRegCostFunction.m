function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = computeCost(X, y, theta);
reg_val = (lambda * sum(theta(2:n) .^ 2)) / (2 * m);

J = h + reg_val;




grad1 = (X' * ((X * theta) - y)) / m;
grad_reg = lambda * [0; theta(2:n)] / m;

grad = grad1 + grad_reg;










% =========================================================================

grad = grad(:);

end
