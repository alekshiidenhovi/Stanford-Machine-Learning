function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Initialize basic vectors and variable m
C_options = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_options = [0.01 0.03 0.1 0.3 1 3 10 30];
m = length(C_options);

predictions = zeros(length(yval), m^2); % indices 1->64

% Loop over C_options and sigma_options and train models for all mxm
% combinations
for i = 1:m
    for j = 1:m 
        model = svmTrain(X, y, C_options(i), @(x1, x2)gaussianKernel(x1, x2, sigma_options(j)));
        predictions(:, m*(i-1) + j) = svmPredict(model, Xval);
    end
end

% Calculate the error for the predictions of the cross-validation sets.
error = mean(double(predictions ~= yval));    

% Take the index of the model with minimum error
[value, index] = min(error); % indices 1->64

% Return optimal values C and sigma
C = C_options(floor((index - 1) / m) + 1);
sigma = sigma_options(mod(index, m));





% =========================================================================

end
