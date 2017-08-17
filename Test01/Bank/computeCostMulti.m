function J = computeCostMulti(X, y, theta, lambda)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


J =  (sum((y.*log(sigmoid(X*theta))) + ((1-y).*log(1-sigmoid(X*theta))))/m * -1) + (lambda/2/m * sum(theta(2:length(theta)).^2));


% =========================================================================

end
