%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 16;
num_labels = 2;

% Load Training Data
raw_data = csvread('sample.csv');
m = size(raw_data, 1);
n = size(raw_data, 2);

train_data = raw_data(2:0.8*m+1, 1:n-1);
validation_data = raw_data(0.8*m+1:0.9*m, 1:n-1);
test_data = raw_data(0.9*m+1:m, 1:n-1);

train_data = [ones(size(train_data, 1), 1) train_data];
validation_data = [ones(size(validation_data, 1), 1) validation_data];
test_data = [ones(size(test_data, 1), 1) test_data];

train_y = raw_data(2:0.8*m+1, n);
validation_y = raw_data(0.8*m+1:0.9*m, n);
test_y = raw_data(0.9*m+1:m, n);

fprintf('Program paused. Press enter to continue.\n');
pause;


% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

feature_size = n;
lambda = 100;


initial_theta = zeros(feature_size, 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);

[theta_c] = fmincg(@(t)(lrCostFunction(t, train_data, train_y, lambda)), initial_theta, options);
[J grad] = lrCostFunction(theta_c, train_data, train_y, lambda);

fprintf('\nCost: %f\n', J);
%fprintf('Gradients:\n');
%fprintf(' %f \n', grad);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-One ================

%pred = predictOneVsAll(all_theta, X);

res = sigmoid(test_data * theta_c) >=0.5 ;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(res == test_y)) * 100);