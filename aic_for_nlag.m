function [aic]=aic_for_nlag(y,Mdl,numMdl,logL,numParam,numObs)

for j = 1:numMdl
    y0 = y(1:Mdl(j).P);             % Presample
    yest = y((Mdl(j).P+1):end);     % Estimation sample
    [EstMdl,~,logL(j)] = estimate(Mdl(j),yest,Y0=y0, Display="off");
    results = summarize(EstMdl);
    numParam(j) = results.NumEstimatedParameters;
    numObs(j) = results.SampleSize; 
end

aic = aicbic(logL,numParam,numObs,Normalize=true);

end