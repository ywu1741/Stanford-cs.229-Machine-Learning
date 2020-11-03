function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

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

test_param = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
C_vec = test_param;
sigma_vec = test_param;

c = size(C_vec, 1);
s = size(sigma_vec, 1);
error = Inf;

for i = 1:c,
  C_test = C_vec(i);
  for j = 1:s,
    sigma_test = sigma_vec(j);
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    pred = svmPredict(model, Xval);
    error_test = mean(double(pred ~= yval));
    if error_test < error
      error = error_test;
      C = C_test;
      sigma = sigma_test;
    endif
  endfor
endfor


% =========================================================================

end
