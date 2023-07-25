%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  mAt = D2_2(va, nk)
%%
%%  "D2_2" sets matrix lower-triangular matrix with free
%%  elements inserted from 1*na vector "a"
%%
%%  [input]
%%    va:    1*na vector
%%    nk:    # of variables
%%
%%  [output]
%%    mAt:   nk*nk lower-triangular matrix with diagonal
%%           elements equal to one
%%

function mAt = D2_2(va, nk)

mAt = eye(nk);

for i = 2 : nk
    mAt(i, 1:i-1) = va((i-1)*(i-2)/2+1 : i*(i-1)/2);
end
