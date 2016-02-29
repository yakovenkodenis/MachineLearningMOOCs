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

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y, :);

[x_i, x_j] = size(X);
a1 = [ones(x_i, 1), X];

z2 = a1*Theta1';

a2 = sigmoid(z2);

[a2_i, a2_j] = size(a2);
a2 = [ones(a2_i, 1), a2];

z3 = a2*Theta2';

a3 = sigmoid(z3);


sum_part_1 = -y_matrix .* log(a3);

sum_part_2 = (1 - y_matrix) .* log(1 - a3);

J = (1 / m) * sum(sum(sum_part_1 - sum_part_2));



[theta1_i, theta1_j] = size(Theta1);
[theta2_i, theta2_j] = size(Theta2);
reg_multiplier = lambda / (2 * m);
reg_term_1 = sum(sum(Theta1(:, 2:theta1_j) .^ 2));
reg_term_2 = sum(sum(Theta2(:, 2:theta2_j) .^ 2));
regularization = reg_multiplier * (reg_term_1 + reg_term_2);

J = J + regularization;



% -------------------------------------------------------------


d3 = a3 - y_matrix;

d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1 = Theta1 .* (lambda / m);
Theta2 = Theta2 .* (lambda / m);

Theta1_grad = Delta1 .* (1 / m) + Theta1;
Theta2_grad = Delta2 .* (1 / m) + Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
