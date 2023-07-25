function [iso_code Country Year CO2 CO2_percapita Population GDP Country_set Year_set Country_judge] = data_import(file_path)

global iso_code Country Year CO2 CO2_percapita Population GDP Country_set Year_set Country_judge;
clc;
%cd path
file_id = fopen(file_path);
C = textscan(file_id, '%s%s%d%f%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
iso_code = string(C{1,1});
Country = string(C{1,2});
Year = C{1,3};
CO2 = C{1,4};
CO2_percapita = C{1,5};
Population = C{1,6};
GDP = C{1,7};

%retrieve the set of countries
Country_set = unique(Country);
%retrieve the set of years
Year_set = unique(Year);
%determine if the record is a country (1 as country, 0 as region)
Country_judge = iso_code ~= "";