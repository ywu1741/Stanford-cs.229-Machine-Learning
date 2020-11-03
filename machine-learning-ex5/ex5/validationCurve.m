function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


% Useful variables
t = size(X, 1);
n = size(Xval, 1);

% validation curve

for i = 1:length(lambda_vec),
  theta = trainLinearReg(X, y, lambda_vec(i));
  J_train = (1/(2*t))*(ones(1,t)*(X*theta-y).^2);
  error_train(i,1) = J_train;
  J_cv = (1/(2*n))*(ones(1,n)*(Xval*theta-yval).^2);
  error_val(i,1) = J_cv;
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)

%


% =========================================================================

end
