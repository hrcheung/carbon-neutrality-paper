%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  [] = D2_9(nsim)
%%
%%  "D2_9" implements D2_9 estimation for TVP-VAR model
%%
%%  [input]
%%      nsim:  # of D2_9 iterations
%%

function [CO2_2050, GDP_2050] = mote_carlo_simulation(nsim, CO2_dt,GDP_dt, tag)
country_tag = ['US'; 'UK'; 'JP'; 'CN'];
% country_tag = ['美国'; '英国'; '日本'; '中国'];
%碳中和值
CO2_tag = [5113.455*0.2; 600.344*0.2; 1158.007*0.2; 5876.555*0.5];
%mod(tag,4) 0 1 2 3
%nsim=10000;
%CO2_dt = JAPAN_CO2;
%GDP_dt = JAPAN_GDP;
%tag = 2;
global m_my m_asvar m_nl m_ns m_nk m_fli m_flSb m_nimp m_flfi ...
       m_iseed m_dvb0 m_dVb0 m_dva0 m_dVa0 m_dvh0 m_dVh0 m_k
   
tic;

%%--- set default options ---%%

if isempty(m_fli) == 1
    m_fli = 0;
end
if isempty(m_flSb) == 1
    m_flSb = 0;
end
if isempty(m_nimp) == 1
    m_nimp = 12 + 1;
end
if isempty(m_flfi) == 1
    m_flfi = 1;
end
if isempty(m_iseed) == 1
    m_iseed = 1;
end

%初始化随机数种子
%均匀分布随机数
rand('state', m_iseed);
%正态分布随机数
randn('state', m_iseed);


%%--- set variables ---%%
%预测30个
T=32;
ns = m_ns + T;  % # of time periods n
nk = m_nk;  % # of series变量数k
nl = m_nl;  % # of lags s
% m_flfi:   fast computing of response (flag)
% beta的待估参数个数
nb = nk * (nk*nl + m_fli);  % # of coefficients in beta
% A的下三角（除对角线为1）剩下的a的待估参数个数
na = nk * (nk-1) / 2;       % # of parameters in a
%有y,延长y的长度30
m_my = [m_my ;zeros(T, nk)];
%存结果
CO2_2050 = zeros(nsim, 1);

if m_fli == 1
    vym = zeros(1, nk);
else
    vym = mean(m_my);
end
m_my = m_my - ones(ns, 1) * vym;

%拟合值
myy = zeros(ns,nk);

myh = zeros(ns, nk);
mya = zeros(ns, nk);
%可能与beta有关
amX = zeros(nk, nb, ns);
%可能与a有关
amXh = zeros(nk, na, ns);
%可能与h有关
amG2 = zeros(nk, nk, ns);
mai = zeros(ns, na);
%此处有y
for i = nl+1 : ns-T
    amX(:, :, i) = setMatrix(m_my(i-nl:i-1, :), m_fli);
end
%每一步的beta均值 18个
mb = zeros(ns, nb);
%每一步的a均值 3个
ma = zeros(ns, na);
%每一步的h均值 3个
mh = zeros(ns, nk);
mh_ = zeros(ns, nk);

%beta,a,h的协方差矩阵
mSigb = eye(nb) * 0.002;
mSiga = eye(na) * 0.005;
mSigh = eye(nk) * 0.005;

vidb = 1 : nb;
if m_fli == 1
    vidi = (0 : nk-1) * (nk*nl+1) + 1;
	vidb(vidi) = [];
end
[v1, v2] = find(triu(reshape(1:nk^2, nk, nk)', 1));
vida = (v1-1)*nk + v2;

%%--- prior ---%%

if isempty(m_dvb0) == 1
  if m_flSb == 1
    m_dvb0 = 25;          % Sigma ~ IW(vb0, I*Vb0)
    m_dVb0 = 1e-4;
  else
    m_dvb0 = 40;          % sigb_i^2 ~ IG(va0/2, Va0/2) 
    m_dVb0 = 2*1e-4;
  end
elseif m_flSb == 0
    m_dvb0 = m_dvb0*2;
    m_dVb0 = m_dVb0*2;
end   
if isempty(m_dva0) == 1
  m_dva0 = 8;             % siga_i^2 ~ IG(va0/2, Va0/2)
  m_dVa0 = 2*1e-4;    
end
if isempty(m_dvh0) == 1
  m_dvh0 = 8;             % sigh_i^2 ~ IG(vh0/2, Vh0/2)
  m_dVh0 = 2*1e-4;    
end

vb0 = zeros(nb, 1);       % b_1 ~ N(b0, Sb0)
mSb0 = eye(nb) * 10;
va0 = zeros(na, 1);       % a_1 ~ N(a0, Sa0)
mSa0 = eye(na) * 10;
vh0 = zeros(nk, 1);       % h_1 ~ N(h0, Sh0)
mSh0 = eye(nk) * 50;

% b_1 ~ N(b0, Sb0)
%{
if mod(tag,4) == 0
    vb0 = [0.091 -0.126 0.379 -0.102 -0.127 0.029 -0.195 -0.117 0.398 -0.404 -0.161 -0.542 -0.203 0.544 0.019 0.392]';
    mSb0 = eye(nb) * 1;
elseif mod(tag,4) == 1
    vb0 = [-0.483 0.058 -0.218 0.672]';
    mSb0 = eye(nb) * 1;
elseif mod(tag,4) == 2
    vb0 = [0.296 0.161 0.028 -0.071 0.525 -0.121 0.233 0.094 0.296 0.161 0.028 -0.071 0.525 -0.121 0.233 0.094]';
    mSb0 = eye(nb) * 1;    
end
va0 = zeros(na, 1);       % a_1 ~ N(a0, Sa0)
mSa0 = eye(na) * 1;
vh0 = zeros(nk, 1);       % h_1 ~ N(h0, Sh0)
mSh0 = eye(nk) * 5;
%}
mS0 = eye(nb) * m_dVb0;
dnub = m_dvb0 + ns - nl - 1 -30;
dnua = m_dva0 + ns - nl - 1 -30;
dnuh = m_dvh0 + ns - nl - 1 -30;

    
%%--- set sampling option ---%%

nburn = 0.1 * nsim;         % burn-in period
npmt = 5;                   % # of parameter to store
msamp    = zeros(nsim, npmt);  % sample box
%h的均值和标准差
msamph   = zeros(ns - T, nk);
msamphs  = zeros(ns - T, nk);
%a的均值和标准差
msampa   = zeros(ns - T, na);
msampas  = zeros(ns - T, na);
%inv(a)的均值和标准差
msampai  = zeros(ns - T, na);
msampais = zeros(ns - T, na);
if m_fli == 1
    msampi  = zeros(ns - T, nk);
    msampis = zeros(ns - T, nk);
end
%beta均值和标准差
if m_flfi == 1
    msampb = zeros(ns - T, length(vidb));
else
    mimpm = zeros((ns - T)*m_nimp, nk^2);
end
nK = floor(m_ns/30)-1;      % # of blocks for sampling h


%%--- D2_9 sampling ---%%

fprintf('\nIteration:\n');

%%------------- S A M P L I N G   S T A R T --------------%%

for m_k = -nburn : nsim

  %%--- sampling beta ---%%

    for i = nl+1 : ns-T
        %把a转成A，算奇异矩阵A逆
        mAinv = singular_inv(setLowerTriaMat(ma(i, :), nk));
        %inv(A)*sigma*sigma*inv(A)'
        amG2(:, :, i) = mAinv * diag(exp(mh(i,:))) * mAinv';
        %inv(A)
        mai(i, :) = mAinv(vida)';
    end
  %有y 
    mb(nl+1:ns-T, :) ...
     = simulationSmoother(m_my(nl+1:ns-T,:), amX(:,:,nl+1:ns-T), ...
               amG2(:,:,nl+1:ns-T), mSigb, vb0, mSb0)';

    
  %%--- sampling a ---%%
    %有y
    for i = nl+1 : ns-T
        %y尖
       myh(i, :) = m_my(i, :) - mb(i, :) * amX(:, :, i)';
       %由y尖导出Xt尖
       amXh(:, :, i) = setXHat(myh(i, :), nk, na);
       %sigam
       amG2(:, :, i) = diag(exp(mh(i, :)));
    end
  
    ma(nl+1:ns-T, :) ...
     = simulationSmoother(myh(nl+1:ns-T,:), amXh(:,:,nl+1:ns-T), ...
               amG2(:,:,nl+1:ns-T), mSiga, va0, mSa0)';
  
  %%--- sampling h ---%%

    for i = nl+1 : ns-T
        %算At*yt尖
        mya(i, :) = myh(i, :) * setLowerTriaMat(ma(i, :), nk)';
    end
           
    for i = 1 : nk
        mh(nl+1:ns-T, i) ...
         = D2_12(mya(nl+1:ns-T,i), mh(nl+1:ns-T,i), ...
                  mSigh(i,i), vh0(i), mSh0(i,i), nK);
    end


  %%--- sampling Sigma ---%%
  %先验是逆伽马吗，sampling beta、a、h的协方差矩阵
    mdif = diff(mb(nl+1:ns-T, :));
    if m_flSb == 1
      mSb = inv(mS0 + mdif'*mdif);
      mSb = (mSb + mSb')/2;
      [mL, p] = chol(mSb, 'lower');
      if p > 0
        mSb = diag(diag(mSb));
      end
      mSigb = inv(wishrnd(mSb, dnub));
      mSigb = (mSigb + mSigb')/2;
    else
      vSb = m_dVb0 + sum(mdif.^2);
      mSigb = diag(1 ./ gamrnd(dnub/2, 2./vSb));
    end
    
    vSa = m_dVa0 + sum(diff(ma(nl+1:ns-T, :)).^2);
    mSiga = diag(1 ./ gamrnd(dnua/2, 2./vSa));
    
    vSh = m_dVh0 + sum(diff(mh(nl+1:ns-T, :)).^2);
    mSigh = diag(1 ./ gamrnd(dnuh/2, 2./vSh));
    %对应D(61:118)
    %算累积的拟合值
    if m_k >= 1
        for i = 1:m_ns
            myy(i, :) = myy(i, :)+mb(i, :)*amX(:, :, i)' ;
            cc = mb(i, :)*amX(:, :, i)';
            CO2_2050(m_k,i) = cc(1);            
            GDP_2050(m_k,i) = cc(2);
        end
    end
  %%--- sampling y ---%%
    %只需对延长的y进行sampling
   if m_k >= 1
    for i = ns-T+1 : ns     
        %生成Xt
        amX(:, :, i) = setMatrix(m_my(i-nl:i-1, :), m_fli);
        %用上一步y
        
        %beta    + mvnrnd(zeros(nb,1),mSigb,1)      
        mb(i, :) = mb(i-1, :) + mvnrnd(zeros(nb,1),mSigb,1);        
        %a   + mvnrnd(zeros(na,1),mSiga,1)
        ma(i, :) = ma(i-1, :) + mvnrnd(zeros(na,1),mSiga,1);        
        %h      + mvnrnd(zeros(nk,1),mSigh,1)
        mh(i, :) = mh(i-1, :) + mvnrnd(zeros(nk,1),mSigh,1);
        %无信息了不更新sigma  
        %需不需要去更新sigma?
        %y  
        %eta_t随机数
        rndy = mvnrnd(zeros(nk,1),eye(nk),1);        
        %ma(i,:) = mean(ma);
        %mb(i,:) = mean(mb);
        %每年check预测值，若为负数则为0？
        %预测值部分累积求和
        myy(i,:) = myy(i, :) + mb(i, :)*amX(:, :, i)' +  rndy* inv(setLowerTriaMat(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        %单步预测值
        m_my(i, :) = mb(i, :)*amX(:, :, i)' + rndy * inv(setLowerTriaMat(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        cc(:) = mb(i, :)*amX(:, :, i)' + rndy * inv(setLowerTriaMat(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        CO2_2050(m_k,i) = cc(1);
        GDP_2050(m_k,i) = cc(2);
    end       
   end


%%--- storing sample ---%%

    if m_k > 0
        %要画的参数
        %sigma_b前两个对角元素，sigma_a的第一个对角元素,sigma_h的第一个对角元素
        msamp(m_k, :) = [mSigb(1, 1) mSigb(2, 2) ...
                         mSiga(1, 1) ...
                         mSigh(1, 1) mSigh(2, 2)];
        %h加和
        msamph   = msamph  + mh(1:ns - T, :);
        %h方加和
        msamphs  = msamphs + mh(1:ns - T, :).^2;
        %a加和
        msampa   = msampa  + ma(1:ns - T, :);
        %a方加和
        msampas  = msampas + ma(1:ns - T, :).^2;
        %a逆加和
        msampai  = msampai  + mai(1:ns - T, :);
        %a逆方加和
        msampais = msampais + mai(1:ns - T, :).^2;
        
        if m_fli == 1
            msampi  = msampi + mb(1:ns - T, vidi);
            msampis = msampis + mb(1:ns - T, vidi).^2;
        end
        if m_flfi == 1
            msampb = msampb + mb(1:ns - T, vidb);
            
      %%--- D2_8 response ---%%
      %脉冲相应函数
        else
          mimpm = mimpm ...
                + D2_8(nl, m_nimp, mb(1:ns - T, vidb), ma(1:ns - T, :), mh(1:ns - T, :));
        end
        
    end
    %迭代次数计数    
    if mod(m_k, 1000) == 0       % print counter
        fprintf('%i \n', m_k);
    end
end 
%nsim步取平均
myy = myy/nsim;
myy(:,2) = 1000000000*myy(:,2);
m_my(:,2) = 1000000000*m_my(:,2);
if tag ~= 5
%CO2    
%画1966~2018差分的真实b和拟合r以及残差g,2019~2050的预测r
plot(1966:2018,myy(1:m_ns,1),'r')
hold on
plot(1966:2018,m_my(1:m_ns,1),'b')
hold on
plot(1966:2018,m_my(1:m_ns,1)-myy(1:m_ns,1),'g')
hold on
plot(2019:2050,m_my(m_ns + 1: m_ns + T,1),'r')
legend('Fitted Value','Real Value','Residual','Predicted Value')
% legend('拟合值','真实值','残差','预测值')
%画差分还原图
figure()
plot(1966:2018,CO2_dt(62:114),'r')
hold on
%还原1966~2018的拟合值，2019~2050的预测值
for i = 1:m_ns
    myyy(i,1) = myy(i,1) + CO2_dt(m_ns + nl + 3 + i);
    CO2_2050(:,i) =  CO2_2050(:,i) + CO2_dt(m_ns + i +3 +nl);
end
for i = m_ns + 1:m_ns + T
    myyy(i,1) = myy(i,1) + myyy(i-1,1);
    CO2_2050(:,i) =  CO2_2050(:,i) + CO2_2050(:,i-1);
end
plot(1966:2050,myyy(1:m_ns+T,1),'b')
hold on
%画置信区间
%{
for i = 1:ns
    CO2_2050(:,i) = sort(CO2_2050(:,i));
end
hold on
plot(1966:2018 + T,CO2_2050(0.975*nsim,1:m_ns + T),'g--')
hold on
plot(1966:2018 + T,CO2_2050(0.025*nsim,1:m_ns + T),'y--')
hold on
plot(1966:2018 + T,CO2_tag(mod(tag,4)+1)*ones(ns,1),'k.')
%}
xlabel('Year')
% xlabel('年份')
ylabel('CO2')

title([country_tag(mod(tag,4)+1,:) 'CO2 Predicted Curve'])
% title([country_tag(mod(tag,4)+1,:) 'CO2预测曲线'])
legend('Real Value','Predicted Value')
% legend('真实值','预测值')%,'置信上限','置信下限','碳中和指标'

%GDP    
%画1966~2018差分的真实b和拟合r以及残差g,2019~2050的预测r
figure();
plot(1966:2018,myy(1:m_ns,2),'r')
hold on
plot(1966:2018,m_my(1:m_ns,2),'b')
hold on
plot(1966:2018,m_my(1:m_ns,2)-myy(1:m_ns,2),'g')
hold on
plot(2019:2050,m_my(m_ns + 1: m_ns + T,2),'r')
legend('Fitted Value','Real Value','Residual','Predicted Value')
% legend('拟合值','真实值','残差','预测值')
%画差分还原图
figure()
plot(1966:2018,GDP_dt(62:114),'r')
hold on
%还原1966~2018的拟合值，2019~2050的预测值
for i = 1:m_ns
    myyy(i,2) = myy(i,2) + GDP_dt(m_ns + nl + 3 + i);
    GDP_2050(:,i) =  GDP_2050(:,i) + GDP_dt(m_ns + i +3 +nl);
end
for i = m_ns + 1:m_ns + T
    myyy(i,2) = myy(i,2) + myyy(i-1,2);
    GDP_2050(:,i) =  GDP_2050(:,i) + GDP_2050(:,i-1);
end
plot(1966:2050,myyy(1:m_ns+T,2),'b')
hold on
%画置信区间

for i = 1:ns
    GDP_2050(:,i) = sort(GDP_2050(:,i));
end
hold on
plot(1966:2018,GDP_2050(0.975*nsim,1:m_ns),'g--')
hold on
plot(1966:2018,GDP_2050(0.025*nsim,1:m_ns),'y--')

xlabel('Year')
ylabel('GDP')

title([country_tag(mod(tag,4)+1,:) 'GDP Predicted Curve'])
% legend('真实值','预测值','置信上限','置信下限')
legend('Real Value','Predicted Value','Upper Confidence Limit','Lower Confidence Limit')
end
%%--------------- S A M P L I N G   E N D ----------------%%

%%--- output result ---%%

iBm = min([500, nsim/2]);   % bandwidth
iacf = iBm;
%参数名
aspar = char('sigma_b1  ', 'sigma_b2', 'sigma_a1', 'sigma_h1', 'sigma_h2');
aspar2 = char('  s_{b1}', '  s_{b2}', '  s_{a1}', ...
              '  s_{h1}', '  s_{h2}');
    
    
fprintf('\n\n                        [ESTIMATION RESULT]')
fprintf('\n----------------------------------')
fprintf('------------------------------------')
fprintf('\nParameter   Mean      Stdev       ')
fprintf('95%%U       95%%L    Geweke     Inef.')
fprintf('\n----------------------------------')
fprintf('------------------------------------\n')
%sigma^2-->sigma
msamp = sqrt(msamp);
%要画的npmt个参数
for i = 1 : npmt
    %先取出第i个参数
    vsamp = msamp(:, i);
    %排序以方便算置信区间（双边）
    vsamp_s = sort(vsamp);
fprintf('%s %10.4f %10.4f %10.4f %10.4f %9.3f %9.2f\n',...
        aspar(i, :), ...
        [mean(vsamp), std(vsamp), ...
         vsamp_s(floor(nsim*[0.025;0.975]))'], ...
         computeGeweke(vsamp, iBm), ...
         timeSeriesParzen(vsamp, iBm)/var(vsamp))
end          


fprintf('-----------------------------------')
fprintf('-----------------------------------')
fprintf('\nTVP-VAR model (Lag = %i', nl)
fprintf(')\nIteration: %i', nsim)
%假设sigma_b为对角矩阵
if m_flSb == 0
  fprintf('\nSigma(b): Diagonal')
end


%%--- output graphs ---%%
asl = cell(2, nk*2+1);
asl{1, 1} = 'Posterior:';
asl{2, 1} = 'Variable:';
asl{1, 2} = 'Mean';
asl{1, nk+2} = 'Standard deviation';
ii = 2;
for i = 1 : 2
for j = 1 : nk
  asl{2, ii} = char(m_asvar(j));
  ii = ii + 1;
end
end
asm = cell(2, na*2+1);
asm{1, 1} = 'Posterior:';
asm{2, 1} = 'Variable:';
asm{1, 2} = 'Mean';
asm{1, na+2} = 'Standard deviation';
ii = 2;
for i = 1 : 2
for j = 1 : na
  asm{2, ii} = num2str(j);
  ii = ii + 1;
end
end
  
%% parameters %%
figure()
vacf = zeros(iacf, 1);
for i = 1 : npmt
    for j = 1 : iacf
        macf = corrcoef(msamp(j+1:end, i), ...
                           msamp(1:end-j, i));
        vacf(j) = macf(2, 1);
    end
    subplot(3, npmt, i)        
    sysh = stem(vacf);              % autocorrelation
    set(sysh, 'Marker', 'none')
    axis([0 iacf -1 1])
    title(aspar2(i, :))
    subplot(3, npmt, npmt+i);
    plot(msamp(:, i))               % sample path
    title(aspar2(i, :))
    vax = axis;
    axis([0 nsim vax(3:4)])
    subplot(3, npmt, npmt*2+i)
    hist(msamp(:, i), 15)           % posterior density
    title(aspar2(i, :))
end

%% draw h %%

msamph = msamph / nsim;   % posterior mean
msamphs = sqrt(msamphs/nsim - msamph.^2);
                          % posterior standard deviation  
%有y
if m_fli == 1
    m_my = m_my + ones(ns, 1) * vym;
end

figure
for i = 1 : nk
    subplot(2, nk, i);
    plot(m_my(nl+1:end, i))
    vax = axis;
    axis([0 ns-nl vax(3:4)])
    if vax(3) * vax(4) < 0
        line([0, ns], [0, 0], 'Color', ones(1, 3)*0.6)
    end
    if i == 1
      title(['Data: ', char(m_asvar(i))], ... 
            'interpreter', 'latex')
    else
      title(char(m_asvar(i)), 'interpreter', 'latex')
    end
end
for i = 1 : nk
    subplot(2, nk, i+nk);
    plot(exp(msamph(nl+1:end, i)))
    hold on
    plot(exp(msamph(nl+1:end, i) - msamphs(nl+1:end, i)), 'r:')
    plot(exp(msamph(nl+1:end, i) + msamphs(nl+1:end, i)), 'r:')
    hold off
    vax = axis;
    axis([1 ns-nl vax(3:4)])
    if i == 1
      title(['SV $\sigma_t^2=\exp(h_t)$: ', ...
             char(m_asvar(i))], 'interpreter', 'latex')
      legend('Posterior mean', '1SD bands')
    else
      title(char(m_asvar(i)), 'interpreter', 'latex')        
    end
end

mout = [msamph msamphs];
mout(1:nl, :) = NaN(nl, nk*2);

if isequal(exist('tvpvar_vol.xlsx', 'file'), 2)
  delete('tvpvar_vol.xlsx');
end
%%%%%%%xlswrite('tvpvar_vol.xlsx', asl, 'Sheet1', 'A1');
%%%%%%%xlswrite('tvpvar_vol.xlsx', [(1:ns - T)', mout], 'Sheet1', 'A3');

%% draw a %%

msampa = msampa / nsim;   % posterior mean
msampas = sqrt(msampas/nsim - msampa.^2);
                          % posterior standard deviation  
figure
for i = 1 : na
    subplot(ceil(na/2), 2, i);
    plot(msampa(nl+1:end, i))
    hold on
    plot(msampa(nl+1:end, i) - msampas(nl+1:end, i), 'r:')
    plot(msampa(nl+1:end, i) + msampas(nl+1:end, i), 'r:')
    hold off
    vax = axis;
    axis([0 ns-nl vax(3:4)])
    if vax(3) * vax(4) < 0
        line([nl-1, ns+1], [0, 0], 'Color', ones(1, 3)*0.6)
    end
    title(['$a_{', num2str(i), 't}$'], ...
          'interpreter', 'latex')
    if i == 1
      legend('Posterior mean', '1SD bands')
    end
end

mout = [msampa msampas];
mout(1:nl, :) = NaN(nl, na*2);

if isequal(exist('tvpvar_a.xlsx', 'file'), 2)
  delete('tvpvar_a.xlsx');
end
%%%%%%%xlswrite('tvpvar_a.xlsx', asm, 'Sheet1', 'A1');
%%%%%%%xlswrite('tvpvar_a.xlsx', [(1:ns - T)', mout], 'Sheet1', 'A3');

%% draw a-inverse %%

msampai = msampai / nsim;   % posterior mean
msampais = sqrt(msampais/nsim - msampai.^2);
                          % posterior standard deviation  
figure
for i = 1 : na
    subplot(ceil(na/2), 2, i);
    plot(msampai(nl+1:end, i))
    hold on
    plot(msampai(nl+1:end, i) - msampais(nl+1:end, i), 'r:')
    plot(msampai(nl+1:end, i) + msampais(nl+1:end, i), 'r:')
    hold off
    vax = axis;
    axis([0 ns-nl vax(3:4)])
    if vax(3) * vax(4) < 0
        line([nl-1, ns+1], [0, 0], 'Color', ones(1, 3)*0.6)
    end
    if i == 1
      title(['$\tilde{a}_{', num2str(i), ...
             't}$ ($A_t^{-1}$: ', ...
             char(m_asvar(fix((vida(i)-1)/nk)+1)), ...
             '$\to$', ...
             char(m_asvar(mod(vida(i)-1,nk)+1)), ')'], ...
             'interpreter', 'latex')
      legend('Posterior mean', '1SD bands')
    else
      title(['$\tilde{a}_{', num2str(i), 't}$ (', ...
             char(m_asvar(fix((vida(i)-1)/nk)+1)), ...
             '$\to$', ...
             char(m_asvar(mod(vida(i)-1,nk)+1)), ')'], ...
             'interpreter', 'latex')
    end
end

mout = [msampai msampais];
mout(1:nl, :) = NaN(nl, na*2);

if isequal(exist('tvpvar_ai.xlsx', 'file'), 2)
  delete('tvpvar_ai.xlsx');
end
%%%%%%%xlswrite('tvpvar_ai.xlsx', asm, 'Sheet1', 'A1');
%%%%%%%xlswrite('tvpvar_ai.xlsx', [(1:ns - T)', mout], 'Sheet1', 'A3');

if m_fli == 1
    
  %% draw intercept %%

  msampi = msampi / nsim;   % posterior mean
  msampis = sqrt(msampis/nsim - msampi.^2);
                          % posterior standard deviation  

  figure
  for i = 1 : nk
    subplot(ceil(nk/2), 2, i);
    plot(msampi(nl+1:end, i))
    hold on
    plot(msampi(nl+1:end, i) - msampis(nl+1:end, i), 'r:')
    plot(msampi(nl+1:end, i) + msampis(nl+1:end, i), 'r:')
    hold off
    vax = axis;
    axis([0 ns-nl vax(3:4)])
    if vax(3) * vax(4) < 0
        line([nl-1, ns+1], [0, 0], 'Color', ones(1, 3)*0.6)
    end
    if i == 1
      title(['Intercept ($c_t$): ', char(m_asvar(i))], ...
             'interpreter', 'latex')
      legend('Posterior mean', '1SD bands')
    else
      title(char(m_asvar(i)), 'interpreter', 'latex')        
    end
  end

  mout = [msampi msampis];
  mout(1:nl, :) = NaN(nl, nk*2);
  
  if isequal(exist('tvpvar_int.xlsx', 'file'), 2)
    delete('tvpvar_int.xlsx');
  end
  %%%%%%%xlswrite('tvpvar_int.xlsx', asl, 'Sheet1', 'A1');
  %%%%%%%xlswrite('tvpvar_int.xlsx', [(1:ns - T)', mout], 'Sheet1', 'A3');
end

%% save D2_8 response %%

if m_flfi == 1
    mimpm = D2_8(nl, m_nimp, msampb/nsim, msampa,...
                    msamph);
else
    mimpm = mimpm / nsim;
end

mimpm(1:m_nimp*nl, :) = NaN;
mout = [NaN((ns-T)*m_nimp, 1), kron(ones(ns-T, 1), (0:m_nimp-1)')];
mout((0:ns-T-1)*m_nimp+1, 1) = (1 : ns-T)';
asl = cell(3, nk^2+2);
asl{1, 2} = 'Response:';
asl{2, 2} = 'Shock:';
asl{3, 1} = 't';
asl{3, 2} = 'horizon';
ii = 3;
for i = 1 : nk
for j = 1 : nk
  asl{1, ii} = char(m_asvar(j));
  asl{2, ii} = char(m_asvar(i));
  ii = ii + 1;
end
end

if isequal(exist('tvpvar_imp.xlsx', 'file'), 2)
  delete('tvpvar_imp.xlsx');
end
%%%%%%%xlswrite('tvpvar_imp.xlsx', asl, 'Sheet1', 'A1');
%%%%%%%xlswrite('tvpvar_imp.xlsx', [mout, mimpm], 'Sheet1', 'A4');

fprintf('\n\nRanseed: %i', m_iseed);
fprintf('\nTime: %.2f', toc);
fprintf('\n\n')
