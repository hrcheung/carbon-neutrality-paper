%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  dv = D2_3(vx, iBm)
%%
%%  "D2_3" computes Geweke statistics
%%  for convergence check of D2_9 iteration
%%
%%  [input]
%%    vx:   data series (ns*1 vector)
%%    iBm:  bandwidth (scalar)
%%
%%  [output]
%%    dpv:  P-value for convergence
%%

function dpv = computeGeweke(vx, iBm)

ns = length(vx);
n1 = floor(ns * 0.1);
n2 = floor(ns * 0.5);
vx1 = vx(1:n1);
vx2 = vx(n2+1:end);

dm1 = mean(vx1);
dm2 = mean(vx2);
dv1 = timeSeriesParzen(vx1, iBm);
dv2 = timeSeriesParzen(vx2, iBm);

dz = (dm1 - dm2) / sqrt(dv1 / n1 + dv2 / n2);

dpv = 2 * (1 - normcdf(abs(dz)));
