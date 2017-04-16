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
  vishid     = 0.1*randn(numdims, numhid);%Ȩֵ��ʼֵ����,784*1000
  hidbiases  = zeros(1,numhid);%���ز�ƫ��ֵ��ʼ��Ϊ0,1*1000
  visbiases  = zeros(1,numdims);%���Ӳ�ƫ��ֵ��ʼ��Ϊ0

  poshidprobs = zeros(numcases,numhid);%100*1000������batch���򴫲�ʱ��������������
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);%784*1000
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);%Ȩֵ������784*1000
  hidbiasinc = zeros(1,numhid);%ƫ������
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);% �����������򴫲�ʱ�������������ʣ�100*1000*600
end

for epoch = epoch:maxepoch,%�ܹ�����10��
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,%ÿ�ε������б������е�batch
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 
%��RBM�н���k��Gibbs�����ľ����㷨��Щ�����Ƿ���RBM�����ʾ��Gibbs�ֲ��������rbm�½�
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);%100*784
  %data*vishid=100*1000
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));% �������򴫲�ʱ������ڵ��������� p=(h|v)      
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;%784*1000���������w�õģ�������ÿ��Ԫ�ر�ʾ��Ӧ�Ŀ��Ӳ�ڵ��������ڵ�ĳ˻��������˴����������ݶ�Ӧֵ���ۼӣ�
  poshidact   = sum(poshidprobs);%�������ֵ������ͣ�h��ľ�ֵ����
  posvisact = sum(data);%v�������ľ�ֵ����

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);%100*1000��������������01�����˲�����posprods֮����У������ո���ֵ��С���ж�.

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));% �������ʱ�Ŀ��Ӳ�����%100*1000*1000*784=100*784,p=(v|h)    
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    % ������к����������򴫲������������ֵ  
  negprods  = negdata'*neghidprobs;% ͬ��Ҳ�Ǽ���w��784*1000
  neghidact = sum(neghidprobs);% NEGATIVE PHASE h��ľ�ֵ����
  negvisact = sum(negdata); % NEGATIVE PHASE v��ľ�ֵ����

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));% �ع���Ĳ�ֵ
  errsum = err + errsum;% ����errsumֻ���������ÿ�ε���ʱ��������

   if epoch>5,
     momentum=finalmomentum;%0.5��momentumΪ������һ��Ȩֵ���������ı����������������Խ�٣����������ֵ������΢��һ��
   else
     momentum=initialmomentum;%0.9
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...%vishidinc 784*1000��Ȩֵ����ʱ��������
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
