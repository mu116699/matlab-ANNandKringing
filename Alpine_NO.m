% ʹ������������Կ������յĲ��Խ�����и߶ȵ�һ���ԣ�
% ���Ǵӽ����������kriging�Ĳ��Խ�����и��õ���Ͻ��
clc
clear
%�������������
Interval = 1;
%����RBF���������ز������
Interval_RBF = 50;
%�������ݵ�����
num_train1 = 100;
num_train2 = 100;
num_test   = 1000;
num_sampletotla = num_train1+num_train2;
%ʵ�����ݵ�ά��
dim = 10;
%ͳ�ƴ�ÿ���������е�ʱ��
time_Kring_RBF = zeros(1,1);
time_RBF = zeros(1,1);
time_Kring = zeros(1,1);
%train data1
mid_sampletotal = Interval*lhsdesign(num_sampletotla,dim);
sampletotal = mid_sampletotal';
% samplevalue = sin(sampletotal(1,:)).*sin(sampletotal(2,:)).*sin(sampletotal(3,:))...
%     .*sin(sampletotal(4,:)).*sin(sampletotal(5,:)).*sin(sampletotal(6,:))...
%     .*sin(sampletotal(7,:)).*sin(sampletotal(8,:)).*sin(sampletotal(9,:))...
%     .*sin(sampletotal(10,:)).*sqrt(sampletotal(1,:).*sampletotal(2,:).*...
%     sampletotal(3,:).*sampletotal(4,:).*sampletotal(5,:).*sampletotal(6,:).*...
%     sampletotal(7,:).*sampletotal(8,:).*sampletotal(9,:).*sampletotal(10,:));
samplevalue = sin(sampletotal(1,:)).*sin(sampletotal(2,:)).*sqrt(sampletotal(1,:)...
    .*sampletotal(2,:));
%�������
train_xm_true = mid_sampletotal(1:num_train1,:);
train_y_xm_true = samplevalue(:,1:num_train1);
%train data2
num = num_train1+1;
train_xn_true = mid_sampletotal(num:num_sampletotla,:);
train_y_xn_true = samplevalue(:,num:num_sampletotla);
%test data
test_xx_true  = Interval*lhsdesign(num_test,dim);
%���е�ת������
train_x_xm_true =train_xm_true';
train_x_xn_true =train_xn_true';
test_x_true = test_xx_true';
%ȡ�ñ����е�����
[testx,testy]=size(test_x_true);
%test data for test value for function
% test_y_true =  sin(test_x_true(1,:)).*sin(test_x_true(2,:)).*sin(test_x_true(3,:))...
%     .*sin(test_x_true(4,:)).*sin(test_x_true(5,:)).*sin(test_x_true(6,:))...
%     .*sin(test_x_true(7,:)).*sin(test_x_true(8,:)).*sin(test_x_true(9,:))...
%     .*sin(test_x_true(10,:)).*sqrt(test_x_true(1,:).*test_x_true(2,:).*...
%     test_x_true(3,:).*test_x_true(4,:).*test_x_true(5,:).*test_x_true(6,:).*...
%     test_x_true(7,:).*test_x_true(8,:).*test_x_true(9,:).*test_x_true(10,:));
test_y_true =  sin(test_x_true(1,:)).*sin(test_x_true(2,:)).*sqrt(test_x_true(1,:)...
    .*test_x_true(2,:));
% scatter3(test_x_true(1,:),test_x_true(2,:),test_y_true);
% %ȡ��value��ֵ�����ֵ���й�һ��
% train_y_xm_true_max  =  max(abs(train_y_xm_true));
% train_y_xn_true_max  =  max(abs(train_y_xn_true));
% test_y_true_max      =  max(abs(test_y_true));
%��һ�����ݣ�Ϊ�������Ժ����ṩԭ������д�ӿ�
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
%RBF���������ѵ��
rbf = newrb(train_x_xm, train_y_xm,0,1.0,Interval_RBF);

output1 = rbf(train_x_xm);
output2 = rbf(train_x_xn);
%train data2 + train data1 come from rbf
% train_x_total = [train_x_xn,train_xm];
% train_y_guodu = [output1,train_y_xm];
%����traindata2�����y'
yerror = output1 - train_y_xm;
yerror1 = output2 - train_y_xn;
yerror_e = [yerror,yerror1];
error_y = yerror_e';
%ʹ�����õ��Ħ�y�����˹�ֲ��Ħң�����Ԫ��̬�ֲ���
%ʹ��Kriging�������ģ�ͣ�ʹ�õ�ģ�ͺͶ�Ԫ��̬�ֲ�
%��һ���ġ����ǹ�������������������Ҫ�����ģ�͡�
%������ѵ�����ݼ���train_xn,������Ԥ��ķ�Ӧ��error_y��
%����ʹ��kriging����ѵ�����ģ�ͣ�
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
%����RBF�������Ԥ��ֵ��
RBFKoutput = rbf(test_x);
%����kriging�е����
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
% �������
tic;
train_x = sampletotal;
train_y = samplevalue;
rbf1 = newrb(train_x, train_y,0,1.0,Interval_RBF);
output = rbf1(test_x);
% ����rbf������������
%�������
max_rbf = max(abs(output-test_y));
rmse_rbf = sqrt(sum((output-test_y).^2)/testy);
%����rbf�Ĳ��Ժ�����ͼ��
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
%�������
max_kriging = max(abs(yEst22-test_y));
%plot model
%����kriging�Ĳ��Ե�ͼ��
% % % % figure(3)
% % % % scatter3(test_x(1,:),test_x(2,:),test_y);
% % % % hold on;
% % % % scatter3(test_x(1,:),test_x(2,:),yEst22);
% % % % legend("1","2")
% % % % hold off;
time_Kring(1) = toc;