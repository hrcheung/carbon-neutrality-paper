function [Country_7_new, U_total] = D4_1(Country_7, matrix_cof, N, ceil_CO2)
% ceil_CO2单步政策范围
% 单步的纳什均衡求解,单轮直至Change_u返回值为false
% Change_u存储nash解是否变动，若变则True，否则False
% N距2050剩余年数
%N=30;
U_total = zeros(201, 7);%效用
Country_7_new = Country_7;
[Country_7_new, Change_u] = D4_2(Country_7_new, matrix_cof, N, ceil_CO2);
for j =1 : 7
    U_total(1, j) = Country_7_new(j).Utility;
end
i = 1;%限制死循环
while Change_u  - (i > 50)
    %[Country_7_new, Change_u] = D4_2(Country_7_new, matrix_cof, N, ceil_CO2);%确定中国需提升减排能力
    
    [Country_7_new, Change_u_1] = D4_2(Country_7_new, matrix_cof, N, ceil_CO2);%Nash均衡
    for j =1 : 7
        U_total(2*i , j) = Country_7_new(j).Utility;
    end
    [Country_7_new, Change_u_2] = D4_2(Country_7_new, matrix_cof, N, ceil_CO2);
    for j =1 : 7
        U_total(2*i +1, j) = Country_7_new(j).Utility;
    end
    %[Country_7_new, Change_u_3] = D4_2(Country_7_new, matrix_cof, N, ceil_CO2);
    %for j =1 : 7
    %    U_total(3*i + 1, j) = Country_7_new(j).Utility;
    %end
    Change_u = max([Change_u_1  Change_u_2]);%双0则0 Change_u_3
    disp(['nashInteration:', num2str(i) ,' ',num2str(Change_u_1),' ',num2str(Change_u_2)]);%,' ',num2str(Change_u_3)
    
    %{
    for j =1 : 7
        U_total(i + 1, j) = Country_7_new(j).Utility;
    end
    disp(['nashInteration:', num2str(i),' ',num2str(Change_u)]);     
    %}    
    i = i + 1;
     
end

