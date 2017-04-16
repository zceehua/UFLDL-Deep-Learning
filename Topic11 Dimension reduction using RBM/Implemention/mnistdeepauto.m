% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
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


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

clear all
close all
%����ѵ��rbm���磬rbm���Դ��˲���w��b�ĸ���
%Ȼ�󹹽�decoder��encoder������decoder��encoderʹ����ͬ�Ĳ���w��b
%ʹInput=Output����������Ϊ���ۺ�������backpropgation�͹��������������������������w��b
maxepoch=10; %In the Science paper we use maxepoch=50, but it works just fine. 
numhid=1000; numpen=500; numpen2=250; numopen=30;

fprintf(1,'Converting Raw files into Matlab format \n');
converter; % ת������Ϊmatlab�ĸ�ʽ

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

makebatches;%batchdata=100��784��600����ʾ��600��batch��ÿ��batch����100������
[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm;
hidrecbiases=hidbiases; 
%save filename x y z ��������x��y��z���浽��Ϊfilename.mat�Ķ����Ƶ�����  
save mnistvh vishid hidrecbiases visbiases;

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;%100*1000*600�������������򴫲�ʱ��������������
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthp hidpen penrecbiases hidgenbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2 hidpen2 penrecbiases2 hidgenbiases2;

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);
batchdata=batchposhidprobs;
numhid=numopen; 
restart=1;
rbmhidlinear;
hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
save mnistpo hidtop toprecbiases topgenbiases;

backprop; 

