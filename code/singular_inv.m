%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  mAi = D2_4(mA)
%%
%%  "D2_4" outputs inv(A)
%%  when singular, Moore-Penrose pseudoinverse of matrix
%%
%%  [input]
%%    mA:      nk*nk matrix
%%
%%  [output]
%%    mAi:     nk*nk matrix
%%

function mAi = singular_inv(mA)

drA = rcond(mA);

if isnan(drA) || (drA < eps*10^2)
    mAi = inv(diag(diag(mA)));
else
    mAi = inv(mA);
end