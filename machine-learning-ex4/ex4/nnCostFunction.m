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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


input_layer = [ones(m, 1) X];
one_hot_y = eye(max(y))(y,:);

hidden_layer = [ones(m, 1) sigmoid(input_layer * Theta1')];

h = sigmoid(hidden_layer * Theta2');





%J = sum(sum(-one_hot_y .* log(h) - (1 - one_hot_y) .* log(1 - h))) * (1 / m);

J = (sum(sum((-one_hot_y .* log(h)) - ((1 - one_hot_y) .* log(1 - h)))) / m) + ...
  (lambda * sum(sum(Theta1(:, 2:end) .^2))) / (2 * m) + ...
  (lambda * sum(sum(Theta2(:, 2:end) .^2))) / (2 * m);


% end of part 1


delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));



for i = 1:m
  % take one row at a time
  X_i = input_layer(i, :);
  y_i = one_hot_y(i, :);
  hid_i = hidden_layer(i, :);
  
  h = sigmoid([1 sigmoid(X_i * Theta1')] * Theta2');
  err = h - y_i;
  err2 = err * Theta2 .* sigmoidGradient([1 (X_i * Theta1')]);
  err2_no_bias = err2(2:end);
  
  delta1 = delta1 + (err2_no_bias' * X_i);
  delta2 = delta2 + (err' * [1 sigmoid(X_i * Theta1')]);
endfor




% J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)(:)));

theta1_zero_bias = [ zeros(size(Theta1, 1), 1) Theta1(:, 2:end) ];
theta2_zero_bias = [ zeros(size(Theta2, 1), 1) Theta2(:, 2:end) ];

Theta1_grad = (1 / m) * delta1 + (lambda / m) * theta1_zero_bias;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * theta2_zero_bias;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
