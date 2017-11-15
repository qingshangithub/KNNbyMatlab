clc
testData =load('test_batch.mat');
traidxata1=load('data_batch_1.mat');
traidxata2=load('data_batch_2.mat');
traidxata3=load('data_batch_3.mat');
traidxata4=load('data_batch_4.mat');
traidxata5=load('data_batch_5.mat');

trainImages=double([traidxata1.data;traidxata2.data;traidxata3.data;traidxata4.data;traidxata5.data]');
trainLabels=double([traidxata1.labels;traidxata2.labels;traidxata3.labels;traidxata4.labels;traidxata5.labels]);
testImages=double(testData.data');
testLabels=double(testData.labels);

N = 3072;  
K = 10;

trainLength = length(trainImages);  
testLength = length(testImages);  
testResults = linspace(0,0,length(testImages));  
error=0;  
tic;  
for i=1:testLength  
    tmpImage = repmat(testImages(:,i),1,trainLength);  
    tmpImage = abs(trainImages-tmpImage); 
    comp=sum(tmpImage);  
    [dist,idx] = sort(comp);
    len = min(K,length(dist));
    testResults(i) = mode(trainLabels(idx(1:len)));    
    if (testResults(i) ~= testLabels(i))
    disp(i);
    error=error+1; 
    end
end 
  
fprintf('准确率为：%f\n',1-error/testLength)
toc;  
disp(toc-tic);  