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

X = [ones(m,1) X]; % 5000x401
hidden = sigmoid(X * Theta1'); % 5000x25
hidden = [ones(m, 1) hidden]; % 5000x26
h_x = sigmoid(hidden * Theta2'); % 5000x10

vec_y = zeros(num_labels, m); % 10x5000
for i = 1:m
    vec_y(y(i), i) = 1;
end

% for i = 1:m
%     J = J + 1 / m * sum(-vec_y(i) .* log(h_x(i)) - (1 - vec_y(i)) .* log(1 - h_x(i)));
% end
J = sum(sum(-1 / m * (vec_y' .* log(h_x) + (1 - vec_y') .* log(1 - h_x))));

% regularization
Theta1_nozero = Theta1(:, 2:end);
Theta2_nozero = Theta2(:, 2:end);
J = J + lambda / (2*m) * (sum(sum(Theta1_nozero.^2)) + sum(sum(Theta2_nozero.^2)));


for i = 1:m
    % Part 1: Compute a- and z-values
    a_1 = X(i, :)'; % 401x1
    z_2 = Theta1 * a_1; %25x1
    a_2 = sigmoid(z_2); %25x1
    a_2 = [1; a_2]; %26x1
    z_3 = Theta2 * a_2; %10x1
    a_3 = sigmoid(z_3); %10x1
    
    % Part 2: Set y value checker, compute delta_3.
    y_check = zeros(num_labels, 1); %10x1
    y_check(y(i)) = 1;
    delta_3 = a_3 - y_check; %10x1
    
    % Part 3: Compute the hidden layer (delta_2).
    delta_2 = (Theta2' * delta_3) .* (a_2 .* (1 - a_2)); %26x1
    delta_2 = delta_2(2:end); %25x1
    
    % Compute the unregularized gradients.
    
    Theta1_grad = Theta1_grad + delta_2 * a_1'; %25x401
    Theta2_grad = Theta2_grad + delta_3 * a_2'; %10x26
    
end

% Regularization
T1_reg = Theta1;
T1_reg(:, 1) = 0;
T2_reg = Theta2;
T2_reg(:, 1) = 0;

Theta1_grad = 1 / m * (Theta1_grad + lambda .* T1_reg);
Theta2_grad = 1 / m * (Theta2_grad + lambda .* T2_reg);









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
