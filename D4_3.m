function [CO2_c, U, Change] = D4_3(i, UT, Disc, AL, P, C, GDP, CO2, Invest, Ex, Im, bench, matrix_cof, N, ceil_CO2)
% 第i个国家最优决策
% N距2050剩余年数
% i = 1;
% 约束条件的进一步确定
% 1、2050实现碳中和，每一步的可行集受该约束影响
% 2、CO2排放为正值
% 30年随机生成
% CO2_以百分比变动，有变动限制
Change = false;
CO2_c = CO2(i, :);
CO2s = CO2;
n = 10000;
CO2_ = zeros(n, 30);%7*30的矩阵CO2_f 
% 假设CO2_变动区间为-0.1:0.1
% 贴现30期
steps = 40;%分割数
w = steps:-1:1;%抽样权重
%预测C,1*7矩阵X7*30矩阵
%([1:30]*(k or CO2)+bench(i,2))*bench(i,1)未来30年CO2量    
U_now = zeros(1, n);
for j = 1 : 30
    CO2s(:, j) = sum(CO2(:, 1 : j), 2);
end
CO2_f = (CO2s + bench(:,2)*ones(1,30)).*(bench(:,1)*ones(1,30));%其余六国未来30年CO2排放预测

for indexs = 1 : n % 选最优
    CO2_1(indexs, :) = randsample(linspace(ceil_CO2(i,1), ceil_CO2(i,2), steps), 30, true, w);  %/5五年一调整   
    %CO2修正，保证每年CO2排放量非负
    for j = 1 : 30%/5
        %{
        if (5*sum(CO2_1(indexs,1:j)) + bench(i,2)) < 0
            CO2_1(indexs,j) = 0;%若已经为负，修正为0
        end
        CO2_(indexs,5*j-4:5*j) = CO2_1(indexs,j)*ones(1,5);
        %}
        if (sum(CO2_1(indexs,1:j)) + bench(i,2)) < 0
            CO2_1(indexs,j) = 0;%若已经为负，修正为0
        end
        CO2_(indexs,j) = CO2_1(indexs,j);
    end

    if (sum(CO2_(indexs, 1 : N)) + bench(i,2)) <= bench(i,3)%可以实现2050碳中和
        CC = CO2_(indexs, :);
        for j = 1 : 30
            CO2_(indexs, j) = sum(CC(1 : j));
        end
        %对数CO2
        CO2_f(i, :) = log((CO2_(indexs, :) + bench(i,2)*ones(1,30).*(bench(i,1)*ones(1,30))));
        %未来30天的预测值1*30矩阵C
        C = matrix_cof(i,2:8)*CO2_f + matrix_cof(i,1);
        for j = 1:N
            U_now(indexs) = U_now(indexs) + abs((1+Disc(i))^(-j)*P(i, j + 1)*(C(j)/P(i, j + 1))^(1-AL(i))/(1-AL(i)));
        end
        CO2_(indexs, :) = CC;
    end  
end
%优化准则福利增加且能实现碳中和
[U_max, index] = max(U_now);
U = U_max;
if U > UT(i) && U ~= 0 %
    Change = true;
    CO2_c = CO2_(index, :);
else
    U = UT(i);
end
%{
%CO2的选择发生变动时，
if CO2_(index, 1) ~= CO2(i) && U ~= 0 
    Change = true;
    CO2_c = CO2_(index, :);
end
%}