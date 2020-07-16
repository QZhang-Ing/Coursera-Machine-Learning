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


% FP
% Add ones to the X data matrix
X = [ones(m,1) X];
z2 = X * Theta1' % also Thetal1 * X'
a2 = (sigmoid(z2));

% Add ones to the hidden layer
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2'
a3 = sigmoid(z3)
h_theta = a3 % a3 is our output layer so h(x) == a3

% Initialize the y(k). y(k) should contain only 0 or 1
yVector = zeros(m, num_labels) % 10 * 5000
for i = 1:m
    yVector(i,y(i)) = 1;
end

J = 1/m * sum(sum((-1 * yVector .* log(h_theta)) - (1 - yVector) .* log(1 - h_theta)))

regu = lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)))

J = J + regu



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
%   first time.


% For the input layer, where l=1:
a1 = [ones(m,1) X] % 5000*401

% For the hidden layers, where l=2:
z2 = a1 * Theta1'; % 5000*401 * 401*25 = 5000*25
a2 = [ones(m,1) sigmoid(z2)]; % add a bias 5000*26

z3 = a2 * Theta2'; % 5000*26 * 26*10 = 5000*10
a3 = sigmoid(z3); % output layer h(theta) 5000*10

yy = [1:num_labels]==y; % 5000*10
% For the delta values:
delta_3 = a3 - yy; % 5000*10

% (5000*10 * 25*10) .* 5000*25 = 5000*25
delta_2 = (delta_3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% delta_1 is not calculated because we do not associate error with the input    

% Big delta update
% 25*401 + 25*5000 * 5000*401 
Theta1_grad = Theta1_grad + delta_2' * a1;
% 10*26 + 10*5000 * 5000*26 
Theta2_grad = Theta2_grad + delta_3' * a2; 


Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Regularization

% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end










