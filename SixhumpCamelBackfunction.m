%这个函数的测试结果也是可以的，也可以看出来Kriging_RBF
% 建模系统确实提高了函数的拟合精度
clc
clear
%定义采样的区间
Interval = 2;
%定义RBF神经网络隐藏层的数量
Interval_RBF = 50;
%采样数据的数量
num_train1 = 100;
num_train2 = 100;
num_test   = 1000;
num_sampletotla = num_train1+num_train2;
%实验数据的维数
dim = 2;
%统计从每个程序运行的时间
time_Kring_RBF = zeros(1,1);
time_RBF = zeros(1,1);
time_Kring = zeros(1,1);
%train data1
mid_sampletotal = 2*Interval*lhsdesign(num_sampletotla,dim)-Interval;
sampletotal = mid_sampletotal';
samplevalue = 4*sampletotal(1,:).^2-2.1*sampletotal(1,:).^4+1/3*...
    sampletotal(1,:).^6+sampletotal(1,:).*sampletotal(2,:)...
    -4*sampletotal(2,:).^2+4*sampletotal(2,:).^4;

% 0.5+ (sin(sqrt(sampletotal(1,:).^2+sampletotal(2,:).^2))-0.5)...
%     ./(1+0.001*(sampletotal(1,:).^2+sampletotal(2,:).^2));
%随机采样
train_xm_true = mid_sampletotal(1:num_train1,:);
train_y_xm_true = samplevalue(:,1:num_train1);
%train data2
num = num_train1+1;
train_xn_true = mid_sampletotal(num:num_sampletotla,:);
train_y_xn_true = samplevalue(:,num:num_sampletotla);
%test data
test_xx_true  = 2*Interval*lhsdesign(num_test,dim)-Interval;
%其中的转换变量
train_x_xm_true =train_xm_true';
train_x_xn_true =train_xn_true';
test_x_true = test_xx_true';
%取得变量中的数量
[testx,testy]=size(test_x_true);
%test data for test value for function
test_y_true = 4*test_x_true(1,:).^2-2.1*test_x_true(1,:).^4+1/3*...
    test_x_true(1,:).^6+test_x_true(1,:).*test_x_true(2,:)...
    -4*test_x_true(2,:).^2+4*test_x_true(2,:).^4;
scatter3(test_x_true(1,:),test_x_true(2,:),test_y_true);
% %取得value中值的最大值进行归一化
% train_y_xm_true_max  =  max(abs(train_y_xm_true));
% train_y_xn_true_max  =  max(abs(train_y_xn_true));
% test_y_true_max      =  max(abs(test_y_true));
%归一化数据，为其他测试函数提供原函数书写接口
train_x_xm  =    train_x_xm_true/Interval;
train_y_xm  =    train_y_xm_true;
train_x_xn  =    train_x_xn_true/Interval;
train_y_xn  =    train_y_xn_true;
test_xx     =    test_xx_true/Interval;
test_x      =    test_x_true/Interval;
test_y      =    test_y_true;
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%++++Kriging_RBF Neural Network Calculation Process+++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
disp("Kriging_RBF Neural Network Calculation Process")
tic;
%RBF神经网络进行训练
rbf = newrb(train_x_xm, train_y_xm,0,1.0,Interval_RBF);

output1 = rbf(train_x_xm);
output2 = rbf(train_x_xn);
%train data2 + train data1 come from rbf
% train_x_total = [train_x_xn,train_xm];
% train_y_guodu = [output1,train_y_xm];
%计算traindata2的误差y'
yerror = output1 - train_y_xm;
yerror1 = output2 - train_y_xn;
yerror_e = [yerror,yerror1];
error_y = yerror_e';
%使用求解得到的Δy计算高斯分布的σ；（多元正态分布）
%使用Kriging构建误差模型，使用的模型和多元正态分布
%是一样的。但是构建的误差更符合我们想要的误差模型。
%给定的训练数据集是train_xn,给定的预测的反应是error_y；
%下面使用kriging进行训练误差模型；
grp=fitrgp(mid_sampletotal,error_y);
% [x,y]=meshgrid(linspace(0,1,80)',linspace(0,1,80)');
% X=[x(:) y(:)];
% [ypred,ysd] = predict(grp,X);
% figure;
% hold on;
% surf(x,y,reshape(ypred,80,80));
% hold off;
% figure;
% hold on;
% surf(x,y,reshape(ysd,80,80));
%计算RBF神经网络的预测值；
RBFKoutput = rbf(test_x);
%计算kriging中的误差
error_kriging = predict(grp,test_xx);
testoutput = RBFKoutput - error_kriging';
max_rbf_kriging = max(abs(testoutput-test_y));
rmse_rbf_kriging = sqrt(sum((testoutput-test_y).^2)/testy);
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%++++RBF Neural Network Calculation Process+++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
time_Kring_RBF(1) = toc;
disp("RBF Neural Network Calculation Process")
% test
% 随机采样
tic;
train_x = sampletotal;
train_y = samplevalue;
rbf1 = newrb(train_x, train_y,0,1.0,Interval_RBF);
output = rbf1(test_x);
% 计算rbf的神经网络的误差
%最大的误差
max_rbf = max(abs(output-test_y));
rmse_rbf = sqrt(sum((output-test_y).^2)/testy);
%绘制rbf的测试函数的图形
% % % % figure(2)
% % % % scatter3(test_x(1,:),test_x(2,:),test_y);
% % % % hold on;
% % % % scatter3(test_x(1,:),test_x(2,:),output);
% % % % legend("1","2")
% % % % hold off;
time_RBF(1) = toc;
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%++++Kriging Interpolation Calculation Process++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
disp("Kriging Interpolation Calculation Process")
%test kriging
train_x1 = train_x';
train_y1 = train_y';
test_x1 = test_x';
test_y1 = test_y';
k = fitrgp( train_x1, train_y1);
yEst2 = k.predict(test_x1);
yEst22 = yEst2';
rmse_kriging = sqrt(sum(yEst22-test_y)^2/testy);
%最大的误差
max_kriging = max(abs(yEst22-test_y));
%plot model
%绘制kriging的测试的图像
% % % % figure(3)
% % % % scatter3(test_x(1,:),test_x(2,:),test_y);
% % % % hold on;
% % % % scatter3(test_x(1,:),test_x(2,:),yEst22);
% % % % legend("1","2")
% % % % hold off;
time_Kring(1) = toc;