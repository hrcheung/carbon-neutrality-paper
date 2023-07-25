function [country, Change_u]= D4_2(Country_7, matrix_cof, N, ceil_CO2)
Change_u = false;
country = Country_7;
%国家数目
counts = length(Country_7);
%打乱单国求解顺序
randindex = randperm(counts);
CO2 = zeros(7, 30);
for i = 1 : counts
    for j = 1:counts
        %汇总各国参数
        UT(j) = Country_7(j).Utility;
        Disc(j) = Country_7(j).Discount;
        AL(j) = Country_7(j).Alpha;
        P(j,:) = Country_7(j).Population;
        C(j) = Country_7(j).Consume;
        GDP(j) = Country_7(j).GDP;
        CO2(j, :) = Country_7(j).CO2;
        Invest(j) = Country_7(j).Investment;
        Ex(j) = Country_7(j).Export;
        Im(j) = Country_7(j).Import;
        CH(j) = false;
        bench(j,:) = Country_7(j).CO2_bechmark;
    end
    [country(randindex(i)).CO2, country(randindex(i)).Utility, Change(randindex(i))] ...
        = D4_3(randindex(i), UT, Disc, AL, P, C, GDP, CO2, Invest, Ex, Im, bench, matrix_cof, N, ceil_CO2);
end
%{
Change
country(:).CO2
%}
%Change
%至少有一个国家变动选择

if max(Change) == 1
    Change_u = true;
end


