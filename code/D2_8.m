%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  mimp = D2_8(nl, nlen, mb, ma, mh)
%%
%%  "D2_8" computes time-varying D2_8 response
%%  (using average shock size)
%%
%%  [input]
%%    nl:       # of lags
%%    nlen:     length of response to compute
%%    mb,ma,mh: time-varying parameters
%%
%%  [output]
%%    mimp:  (ns*nlen)*(nk^2) matrix
%%

function mimp = D2_8(nl, nlen, mb, ma, mh)

ns = size(mh, 1);
nk = size(mh, 2);

mimp = zeros(ns*nlen, nk^2);
amOmsq = zeros(nk, nk, ns);
my = zeros(nl+nlen, nk);
mbs = [mb ; (ones(nlen, 1) * mb(ns, :))];

vh = mean(mh(nl+1:end, :));     % average shock size

for i = 1 : nk
  for t = nl+1 : ns  
    if i == 1
       amOmsq(:, :, t) = inv(setLowerTriaMat(ma(t, :), nk)) ...
                       * diag(exp(vh/2));
    end

    my(nl+1, :) = amOmsq(:, i, t)';
    
    for j = nl+2 : nl+nlen
      my(j, :) = mbs(t+j-nl-1,:) * setMatrix(my(j-nl:j-1,:), 0)';
    end
    
    mimp((t-1)*nlen+1:t*nlen, (i-1)*nk+1:i*nk) ...
      = my(nl+1:end, :);
  end
end


		
