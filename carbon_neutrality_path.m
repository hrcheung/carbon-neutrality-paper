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

function [CO2_2050, GDP_2050] = carbon_neutrality_path(nsim, CO2_dt,GDP_dt, tag)
% country_tag = ['美国'; '英国'; '日本'; '中国'];
country_tag = ['US'; 'UK'; 'JP'; 'CN'];
%碳中和值
CO2_tag = [5113.455*0.2; 600.344*0.2; 1158.007*0.2; 5876.555*0.5];
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
    amX(:, :, i) = D2_7(m_my(i-nl:i-1, :), m_fli);
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
%基于VAR估得的参数
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

%碳中和路径计数
counts = 0;
myy_f = zeros(ns,1);
%%--- D2_9 sampling ---%%

fprintf('\nIteration:\n');

%%------------- S A M P L I N G   S T A R T --------------%%

for m_k = -nburn : nsim

  %%--- sampling beta ---%%

    for i = nl+1 : ns-T
        %把a转成A，算奇异矩阵A逆
        mAinv = D2_4(D2_2(ma(i, :), nk));
        %inv(A)*sigma*sigma*inv(A)'
        amG2(:, :, i) = mAinv * diag(exp(mh(i,:))) * mAinv';
        %inv(A)
        mai(i, :) = mAinv(vida)';
    end
  %有y 
    mb(nl+1:ns-T, :) ...
     = D2_11(m_my(nl+1:ns-T,:), amX(:,:,nl+1:ns-T), ...
               amG2(:,:,nl+1:ns-T), mSigb, vb0, mSb0)';

    
  %%--- sampling a ---%%
    %有y
    for i = nl+1 : ns-T
        %y尖
       myh(i, :) = m_my(i, :) - mb(i, :) * amX(:, :, i)';
       %由y尖导出Xt尖
       amXh(:, :, i) = D2_6(myh(i, :), nk, na);
       %sigam
       amG2(:, :, i) = diag(exp(mh(i, :)));
    end
  
    ma(nl+1:ns-T, :) ...
     = D2_11(myh(nl+1:ns-T,:), amXh(:,:,nl+1:ns-T), ...
               amG2(:,:,nl+1:ns-T), mSiga, va0, mSa0)';
  
  %%--- sampling h ---%%

    for i = nl+1 : ns-T
        %算At*yt尖
        mya(i, :) = myh(i, :) * D2_2(ma(i, :), nk)';
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
        amX(:, :, i) = D2_7(m_my(i-nl:i-1, :), m_fli);
        %用上一步y        
    
        mb(i, :) = mb(i-1, :) + mvnrnd(zeros(nb,1),mSigb,1);        
        ma(i, :) = ma(i-1, :) + mvnrnd(zeros(na,1),mSiga,1);        
        mh(i, :) = mh(i-1, :) + mvnrnd(zeros(nk,1),mSigh,1);
        %y  
        %eta_t随机数
        rndy = mvnrnd(zeros(nk,1),eye(nk),1);       

        %预测值部分累积求和
        myy(i,:) = myy(i, :) + mb(i, :)*amX(:, :, i)' +  rndy* inv(D2_2(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        %单步预测值
        m_my(i, :) = mb(i, :)*amX(:, :, i)' + rndy * inv(D2_2(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        cc(:) = mb(i, :)*amX(:, :, i)' + rndy * inv(D2_2(ma(i, :), nk)) * diag(exp(mh(i, :)/2));
        CO2_2050(m_k,i) = cc(1);
        GDP_2050(m_k,i) = cc(2);
    end     

    if (((sum(CO2_2050(m_k,ns - T + 1:ns)) + CO2_dt(114)) <= CO2_tag(mod(tag,4)+1)) + ((sum(CO2_2050(m_k,ns - T + 1:ns))+CO2_dt(114)) > 0)) ==2 %实现碳中和
        counts= counts + 1;
        myy_f = myy_f + m_my(:,1);
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
myy_f = myy_f/counts;
if tag ~= 5
%CO2    
%{
%画1966~2018差分的真实b和拟合r以及残差g,2019~2050的预测r
plot(1966:2018,myy(1:m_ns,1),'r')
hold on
plot(1966:2018,m_my(1:m_ns,1),'b')
hold on
plot(1966:2018,m_my(1:m_ns,1)-myy(1:m_ns,1),'g')
hold on
plot(2019:2050,m_my(m_ns + 1: m_ns + T,1),'r')
legend('拟合值','真实值','残差','预测值')
%画差分还原图
%}
figure()
plot(1966:2018,CO2_dt(62:114),'r')
hold on
%还原1966~2018的拟合值，2019~2050的预测值
for i = 1:m_ns
    myyy(i,1) = myy_f(i) + CO2_dt(m_ns + nl + 3 + i);
    %CO2_2050(:,i) =  CO2_2050(:,i) + CO2_dt(m_ns + i +3 +nl);
end
for i = m_ns + 1:m_ns + T
    myyy(i,1) = myy_f(i) + myyy(i-1);
    %CO2_2050(:,i) =  CO2_2050(:,i) + CO2_2050(:,i-1);
end
%plot(1966:2018,myyy(1:m_ns,1),'r')
%hold on
plot(2018:2050,myyy(m_ns:m_ns+T,1),'b')
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
%}
plot(1966:2018 + T,CO2_tag(mod(tag,4)+1)*ones(ns,1),'k.')

xlabel('Year')
ylabel('CO2')

title([country_tag(mod(tag,4)+1,:) ' Carbon Neutrality Path'])
% title([country_tag(mod(tag,4)+1,:) '碳中和路径'])
legend('History Value','Predicted Path','Carbon Neutrality Target')%,'置信上限','置信下限'
disp(['The probability to achieve neutrality goal：', num2str(counts/nsim)])
%{
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
legend('拟合值','真实值','残差','预测值')
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

xlabel('年份')
ylabel('GDP')

title([country_tag(mod(tag,4)+1,:) 'GDP预测曲线'])
legend('真实值','预测值','置信上限','置信下限')
%}

end

