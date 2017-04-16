% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);%[100,784,600]

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);%权值初始值随便给,784*1000
  hidbiases  = zeros(1,numhid);%隐藏层偏置值初始化为0,1*1000
  visbiases  = zeros(1,numdims);%可视层偏置值初始化为0

  poshidprobs = zeros(numcases,numhid);%100*1000，单个batch正向传播时隐含层的输出概率
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);%784*1000
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);%权值增量，784*1000
  hidbiasinc = zeros(1,numhid);%偏置增量
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);% 整个数据正向传播时隐含层的输出概率，100*1000*600
end

for epoch = epoch:maxepoch,%总共迭代10次
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,%每次迭代都有遍历所有的batch
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 
%在RBM中进行k步Gibbs抽样的具体算法，些样本是符合RBM网络表示的Gibbs分布，详情见rbm章节
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);%100*784
  %data*vishid=100*1000
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));% 样本正向传播时隐含层节点的输出概率 p=(h|v)      
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;%784*1000，这个是求w用的，矩阵中每个元素表示对应的可视层节点和隐含层节点的乘积（包含此次样本的数据对应值的累加）
  poshidact   = sum(poshidprobs);%针对样本值进行求和，h层的均值向量
  posvisact = sum(data);%v层样本的均值向量

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);%100*1000，将隐含层数据01化（此步骤在posprods之后进行），按照概率值大小来判定.

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));% 反向进行时的可视层数据%100*1000*1000*784=100*784,p=(v|h)    
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    % 反向进行后又马上正向传播的隐含层概率值  
  negprods  = negdata'*neghidprobs;% 同理也是计算w，784*1000
  neghidact = sum(neghidprobs);% NEGATIVE PHASE h层的均值向量
  negvisact = sum(negdata); % NEGATIVE PHASE v层的均值向量

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));% 重构后的差值
  errsum = err + errsum;% 变量errsum只是用来输出每次迭代时的误差而已

   if epoch>5,
     momentum=finalmomentum;%0.5，momentum为保持上一次权值更新增量的比例，如果迭代次数越少，则这个比例值可以稍微大一点
   else
     momentum=initialmomentum;%0.9
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...%vishidinc 784*1000，权值更新时的增量；
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
