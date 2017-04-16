function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    
% The input theta is a vector because minFunc only deal with vectors. In
% this step, we will convert theta to matrix format such that they follow
% the notation in the lecture notes.
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Loss and gradient variables (your code needs to compute these values)
m = size(data, 2);%������ĸ���

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the loss for the Sparse Autoencoder and gradients
%                W1grad, W2grad, b1grad, b2grad
%
%  Hint: 1) data(:,i) is the i-th example
%        2) your computation of loss and gradients should match the size
%        above for loss, W1grad, W2grad, b1grad, b2grad

% z2 = W1 * x + b1
% a2 = f(z2)
% z3 = W2 * a2 + b2
% h_Wb = a3 = f(z3)

z2 = W1 * data + repmat(b1, [1, m]);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, [1, m]);
a3 = z3;

rhohats = mean(a2,2);
rho = sparsityParam;
KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));


squares = (a3 - data).^2;
squared_err_J = (1/2) * (1/m) * sum(squares(:));
weight_decay_J = (lambda/2) * (sum(W1(:).^2) + sum(W2(:).^2));
sparsity_J = beta * KLsum;

cost = squared_err_J + weight_decay_J + sparsity_J;%��ʧ����ֵ

% delta3 = -(data - a3) .* fprime(z3);
% but fprime(z3) = a3 * (1-a3)
delta3 = -(data - a3);
beta_term = beta * (- rho ./ rhohats + (1-rho) ./ (1-rhohats));
delta2 = ((W2' * delta3) + repmat(beta_term, [1,m]) ) .* a2 .* (1-a2);

W2grad = (1/m) * delta3 * a2' + lambda * W2;
b2grad = (1/m) * sum(delta3, 2);
W1grad = (1/m) * delta2 * data' + lambda * W1;
b1grad = (1/m) * sum(delta2, 2);





%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

%sigmoid�����������󵼺���
function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end
