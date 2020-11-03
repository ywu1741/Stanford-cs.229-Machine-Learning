function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hx = sigmoid(X*theta);
e1 = sum((-y).*log(hx));
e2 = sum((y-1).*log(1-hx));
si = size(theta)(1);
J = (1/m)*(e1+e2)+(lambda/(2*m))*sum(theta(2:si).*theta(2:si));

grad = (1/m)*X'*(hx-y)+(lambda/m)*theta;
grad(1) = grad(1)-(lambda/m)*theta(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
