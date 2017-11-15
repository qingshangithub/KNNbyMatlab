clc
%导入数据
trainImages = loadMNISTImages('train-images.idx3-ubyte'); 
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');  
testImages = loadMNISTImages('t10k-images.idx3-ubyte');  
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');  
N = 784;  
K = 10; 

%二值化
trainImages=trainImages>0;
testImages=testImages>0;

trainLength = length(trainImages);  
testLength = length(testImages);  
testResults = linspace(0,0,length(testImages));   
error=0; 
tic;  
for i=1:testLength  
    tmpImage = repmat(testImages(:,i),1,trainLength);  
    tmpImage = abs(trainImages-tmpImage); 
    comp = sqrt(sum(tmpImage.^2));
    [dist,idx] = sort(comp);
    len = min(K,length(dist));
    %找出众数
    testResults(i) = mode(trainLabels(idx(1:len)));
    if (testResults(i) ~= testLabels(i))
        disp(i);
        error=error+1; 
    end
end  

fprintf('准确率为：%f\n',1-error/testLength)
toc;  
disp(toc-tic);  