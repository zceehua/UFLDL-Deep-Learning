function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------    
    %% ��Ŀ��Ĵ��ۺ���
    delta = weightMatrix*featureMatrix-patches;
    fResidue = sum(sum(delta.^2))./numExamples;%�ع����
    fWeight = gamma*sum(sum(weightMatrix.^2));%��ֹ����Ԫ��ֵ����
    cost = fResidue+fWeight;
    %���˽ṹ
    %sparsityMatrix = sqrt(groupMatrix*(featureMatrix.^2)+epsilon);
    %fSparsity = lambda*sum(sparsityMatrix(:)); %������ϵ���Եĳͷ�ֵ
    %cost = fResidue+fWeight+fSparsity; %Ŀ��Ĵ��ۺ���
    
   
    
    %% ��Ŀ����ۺ�����ƫ������
    grad = (2*weightMatrix*featureMatrix*featureMatrix'-2*patches*featureMatrix')./numExamples+2*gamma*weightMatrix;
    grad = grad(:);%ת��Ϊ1��
    %a =
    % 1     2     3
    % 4     5     6
    % 7     8     9
    %a(:);=
	% 1
    % 4
    % 7
    % 2
    % 5
    % 8
    % 3
    % 6
    % 9
end