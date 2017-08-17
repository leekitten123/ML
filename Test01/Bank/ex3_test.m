%% Initializationclear ; close all; clc%% Setup the parameters you will use for this part of the exerciseinput_layer_size  = 16;num_labels = 2;% Load Training Dataraw_data = csvread('sample.csv');m = size(raw_data, 1);n = size(raw_data, 2);train_data = raw_data(2:0.8*m+1, 1:n-1);validation_data = raw_data(0.8*m+1:0.9*m, 1:n-1);test_data = raw_data(0.9*m+1:m, 1:n-1);train_data = [ones(size(train_data, 1), 1) train_data];validation_data = [ones(size(validation_data, 1), 1) validation_data];test_data = [ones(size(test_data, 1), 1) test_data];train_y = raw_data(2:0.8*m+1, n);validation_y = raw_data(0.8*m+1:0.9*m, n);test_y = raw_data(0.9*m+1:m, n);fprintf('Program paused. Press enter to continue.\n');pause;% Test case for lrCostFunctionfprintf('\nTesting lrCostFunction() with regularization');feature_size = n;initial_theta = zeros(feature_size, 1);%%%%%%%%%%lambda = 0.1:0.1:5;iter = size(lambda, 2);theta_c = zeros(feature_size, iter);J = zeros(iter, 1);%%%%%%%%%%for i = 1:iteroptions = optimset('GradObj', 'on', 'MaxIter', 50);[theta_c(:, i)] = fmincg(@(t)(lrCostFunction(t, train_data, train_y, lambda(i))), initial_theta, options);%%%%%%%%%%[J(i)] = lrCostFunction(theta_c(:, i), train_data, train_y, lambda(i));fprintf('Cost: %f\n', J(i));%%%%%%%%%%endfigure;plot(lambda(1:iter), J(1:iter), '-b', 'LineWidth', 2);%fprintf('Gradients:\n');%fprintf(' %f \n', grad);fprintf('Program paused. Press enter to continue.\n');pause;%% ================ Part 3: Predict for One-Vs-One ================%pred = predictOneVsAll(all_theta, X);%%%%%%%%%%accuracy = zeros(1, iter);%%%%%%%%%%res = sigmoid(test_data * theta_c) >= 0.5 ;%fprintf('\nTraining Set Accuracy: %f\n', mean(double(res == test_y)) * 100);%accumarray =accuracy = mean(double(res == test_y)) * 100figure;plot(lambda(1:iter), accuracy(1:iter), '-b', 'LineWidth', 2);