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

       
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Setup some useful variables
m = size(X, 1);

% Turning each y observation into a vector of k units
k = size(unique(y),1);
tst = zeros(m,k);
for i=1:m,
  tst(i,y(i))=1;
endfor
         
         
% Getting h(x)
X = [ones(m, 1) X];
p = zeros(size(X, 1), 1);

z2 = X*Theta1';
ze2 = 1+e.^(-z2);
a2 = 1./ze2;
a2 = [ones(m, 1) a2];

z3 = a2*Theta2';
ze3 = 1+e.^(-z3);
a3 = 1./ze3;

% Calculating the cost 
cst = zeros(m,1);
for i=1:m,
  cst(i) = sum(-tst(i,:).*log(a3(i,:))-(1-tst(i,:)).*log(1-a3(i,:)));
endfor
J=sum(cst)/m

% Incorporating the regularization term
the1_row = size(Theta1, 1);
the1_col = size(Theta1, 2);
the1 = Theta1(1:the1_row,2:the1_col);

the2_row = size(Theta2, 1);
the2_col = size(Theta2, 2);
the2 = Theta2(1:the2_row,2:the2_col);

reg = sum(sum(the1.*the1))+sum(sum(the2.*the2));
reg = lambda*reg/(2*m);

J = J + reg;

% Implementing Backpropagation

Delta1 = zeros(size(Theta1_grad));
Delta2 = zeros(size(Theta2_grad));
size(X)

for i=1:m,
  del3 = a3(i,:)-tst(i,:);
  s_z2 = z2(i,:);
  s_z2 = sigmoidGradient([1 s_z2])';
  del2 = Theta2'*del3'.*s_z2;
  del2 =del2(2:end,:);
  Delta1 = Delta1 + del2*X(i,:);
  Delta2 = Delta2 + del3'*a2(i,:);
endfor

T1_size = size(Theta1,1);
T2_size = size(Theta2,1);
Theta1_grad = (1/m)*Delta1 + (lambda/m)*[zeros(T1_size,1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Delta2 + (lambda/m)*[zeros(T2_size,1) Theta2(:,2:end)];

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
