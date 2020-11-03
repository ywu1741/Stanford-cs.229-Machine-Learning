function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
j = size(theta,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hx = X*theta;
J = (1/(2*m))*ones(1,m)*((hx-y).^2) + (lambda/(2*m))*([0 ones(1,size(theta,1)-1)] * theta.^2);


grad(1) = ((hx-y)'*X(:,1))*(1/m);
for i=2:j,
  grad(i) = ((hx-y)'*X(:,i))*(1/m)+(lambda/m)*theta(i);
endfor



% =========================================================================

grad = grad(:);

end
