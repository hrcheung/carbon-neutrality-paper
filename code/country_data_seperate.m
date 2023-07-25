function [C_CO2,C_CO2_per,C_population,C_GDP] = country_data_seperate(country_name,CO2,Country,Year,CO2_percapita,Population,GDP)

index=Country == country_name & Year >= 1905 & Year <= 2020;
C_CO2=CO2(index);
C_CO2_per=CO2_percapita(index);
C_population=Population(index);
C_GDP=GDP(index);

end