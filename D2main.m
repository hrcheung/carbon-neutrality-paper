if isequal(exist('tvpvar_test.xlsx', 'file'), 2)
  delete('tvpvar_test.xlsx');
end
asvar = {'GDP', 'CO2'};  % 变量名'Population',


xlswrite('tvpvar_test.xlsx', ...
    US_data(1:end-2,:), 'Sheet1', 'A1');
my = xlsread('tvpvar_test.xlsx');
nlag = 4;  % 滞后阶数
D2_10('data', my, asvar, nlag);  % 设置数据
D2_10('fastimp', 1);  % fast computing of response
%[USCO2, USGDP] = D2_9(10000, US_CO2,US_GDP, 0);  %蒙特卡洛模拟10000
[USCO2, USGDP] = D3_1(10000, US_CO2,US_GDP, 0);  %碳中和路径


xlswrite('tvpvar_test.xlsx', UK_data(1:end-2,:), 'Sheet1', 'A1');
my = xlsread('tvpvar_test.xlsx');
nlag = 4;  % 滞后阶数
D2_10('data', my, asvar, nlag);  % 设置数据
D2_10('fastimp', 1);  % fast computing of response
%[UKCO2, UKGDP] = D2_9(10000, UK_CO2, UK_GDP, 1);  %蒙特卡洛模拟10000
[UKCO2, UKGDP] = D3_1(10000, UK_CO2, UK_GDP, 1);  %碳中和路径


xlswrite('tvpvar_test.xlsx', JAPAN_data(1:end-2,:), 'Sheet1', 'A1');
my = xlsread('tvpvar_test.xlsx');
nlag = 4;  % 滞后阶数
D2_10('data', my, asvar, nlag);  % 设置数据
D2_10('fastimp', 1);  % fast computing of response
%[JAPANCO2, JAPANGDP] = D2_9(10000, JAPAN_CO2, JAPAN_GDP, 2);  %蒙特卡洛模拟10000
[JAPANCO2, JAPANGDP] = D3_1(10000, JAPAN_CO2, JAPAN_GDP, 2);  %碳中和路径



xlswrite('tvpvar_test.xlsx', CHINA_data(1:end-2,:), 'Sheet1', 'A1');
my = xlsread('tvpvar_test.xlsx');
nlag = 4;  % 滞后阶数
D2_10('data', my, asvar, nlag);  % 设置数据
D2_10('fastimp', 1);  % fast computing of response
%[CHINACO2, CHINAGDP] = D2_9(10000, CHINA_CO2, CHINA_GDP, 3);  %蒙特卡洛模拟10000
[CHINACO2, CHINAGDP] = D3_1(10000, CHINA_CO2, CHINA_GDP, 3);  %碳中和路径
%D2_1(1, 1);  % draw D2_8 reponse(1)
                         % : 4-,8-,12-period ahead
%D2_1(1, 0);  % draw D2_8 response(2)
                           % : response at t=30,60,90
