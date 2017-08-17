function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, lambda, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

tmp = zeros(size(X, 2), 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    %for i = 1: size(X, 2)
      %tmp(i) = theta(i) - alpha * (sum((X*theta - y).*X(:,i))) / m;
      
      cost = sigmoid(X * theta) - y;
      
      tmp(1) = theta(1) - alpha* X(:, 1)'*cost/m;
      tmp(2:length(tmp)) =  theta(2:length(tmp))- alpha * X(:, 2:size(X, 2))'*cost/m + lambda/m*theta(2:length(theta));
      
    %end

    theta = tmp;
    
    %if mod(num_iters, 10) == 0
      %for j = 1: size(X, 2)
      %  fprintf('[%f] ', theta(j));
      %end
      %fprintf('\n');
    %end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta, lambda);

end

end
