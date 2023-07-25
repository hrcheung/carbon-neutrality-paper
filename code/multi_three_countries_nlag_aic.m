%Multiple Countries: for 3 countries, AIC optimized at nlag=1
Mdl(1)=varm(3,1); 
Mdl(2)=varm(3,2);
Mdl(3)=varm(3,3);
Mdl(4)=varm(3,4);


numMdl = numel(Mdl);
logL = zeros(numMdl,1);      % Preallocate
numParam = zeros(numMdl,1);
numObs = zeros(numMdl,1);

y=[diff(US_CO2),diff(UK_CO2),diff(JAPAN_CO2)];
% y=US_CO2;
for j = 1:numMdl
    y0 = y(1:Mdl(j).P,:);             % Presample
    yest = y((Mdl(j).P+1):end,:);     % Estimation sample
    [EstMdl,~,logL(j)] = estimate(Mdl(j),yest,Y0=y0, ...
        Display="off");
    results = summarize(EstMdl);
    numParam(j) = results.NumEstimatedParameters;
    numObs(j) = results.SampleSize; 
end

aic=aicbic(logL,numParam,numObs,Normalize=true)
[~,minIdx] = min(aic);
Mdl(minIdx).Description