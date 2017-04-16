%% CS294A/CS294W Self-taught Learning Exercise
% 对于自学习来说，我们先用自动编码器来提取未分类样本的特征，得到W1，b1..Wn,bn
% 然后用W1，b1...Wn,bn来求已分类样本在最后一层隐藏层的激活值，然后将激活值送入到分类器中
% 也就是相当于将原来编码器最后一层重构层替换为分类器

%微调表示将之前自动编码器学习到的参数作为神经网络的初始化参数，在已标注样本较多的情况下
%可将已标注样本分为训练集和测试集来继续优化神经网络参数，可得到更好的结果
%一般的，前面的无监督学习到的模型参数可以当做是有监督学习参数的初始化值，
%这样当我们用有大量的标注了的数据时，就可以采用梯度下降等方法来继续优化参数了，
%因为有了刚刚的初始化参数，此时的优化结果一般都能收敛到比较好的局部最优解。
%如果是随机初始化模型的参数值的话，那么在多层神经网络中一般很难收敛到局部较好值，
%因为多层神经网络的系统函数是非凸的。

%那么该什么时候使用微调技术来调整无监督学习的结果呢？只有我们有大量标注的样本下才可以。
%当我们有大量无标注的样本，但有一小部分标注的样本时也是不适合使用微调技术的。
%如果我们不想使用微调技术的话，那么在第三层分类器的设计时，应该采用级联的表达方式，
%也就是说学习到的结果和原始的特征值一起输入。当然了，如果采用了微调技术，则效果更好，
%就不需要继续用级联的特征表达了。
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels.idx1-ubyte');

% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet);

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

opttheta = theta; 
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, loss] = minFunc( @(p) sparseAutoencoderCost(p, ...
      inputSize, hiddenSize, ...
      lambda, sparsityParam, ...
      beta, unlabeledData), ...
      theta, options);









%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels
lambda = 1e-4;
inputSize = hiddenSize;
numClasses = numel(unique(trainLabels));%unique为找出向量中的非重复元素并进行排序

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);









%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);


%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 
