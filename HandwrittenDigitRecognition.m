% This code will try to recognize the handwritten digits using both
% Logestic Regression and Neural Network

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
%% ===================== Loading and Visualizing Data =============
%  We start first by loading and visualizing the dataset.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('HDRdata.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 1: logistic regression implementation ============

% ======================== Part 1.1: COST FUNCTION =======================
% This function computs the cost function using hypothesis and output
% value, and its gradient. 
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ===================== Part 1.2: One-vs-All Training ====================
% Once the cost function and its gradients been computed it can be used to
% calculate the weight parameters (all theta) by minimizing the cost
% function
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 1.3: Predict for One-Vs-All ===================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% =============== Part 2: Neural Network implementation ============

% ================ Part 2.1: Initializing Parameters ================
% In this part instead of using zero matrix for initial weight parameters 
% we will randomize it using (randInitializeWeights.m) function

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% ========================= Part 2.2: Training NN ========================
%  To train neural network, we will use "fmincg", to optimize cost function
%  Recall that this advanced optimizer is able to train our cost functions 
%  efficiently as long as we provide them with the gradient computations.
%  The gradients are calculated in nnCostFunction function.
fprintf('\nTraining Neural Network... \n')

%  Number of iteration is 50
options = optimset('MaxIter', 50);

%  We can also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Part 2.3: Visualize Weights ======================
%  We can now "visualize" what the neural network is learning by displaying
%  the hidden units to see what features they are capturing in the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ====================== Part 2.4: Implement Predict =====================
%  After training the neural network, we would like to use it to predict
%  the labels. The "predict" function is to use the neural network to 
%  predict the labels of the training set. This lets us compute the training
%  set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


