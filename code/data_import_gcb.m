function [year_gcb,emissions,sink,carbon_budget_imbalance,atmos_growth] = data_import_gcb(file_path)

clc;
%cd path
% file_id = fopen(file_path);
% C = textscan(file_id, '%d%f%f%f%f%f%f%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
% year_gcb=C{1,1};
% emissions=C{1,9};
% sink=C{1,10};
% carbon_budget_imbalance=C{1,8};
M=readmatrix(file_path);
year_gcb=M(:,1);
emissions=M(:,9);
atmos_growth=M(:,4);
sink=M(:,10);
carbon_budget_imbalance=M(:,8);