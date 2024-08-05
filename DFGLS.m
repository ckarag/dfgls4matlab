function out = DFGLS(y,varargin)

%Unit root testing using the DF-GLS test of Elliott, Rothenberg & Stock (ERS; 1996),
%with 3 different methods for choosing the optimal lag-length in the underlying ADF regression.
%1. Applies the GLS tranformation (de-mean/de-trend) on yt, yielding the transformed series Yt. 
%2. Runs the (A)DF OLS regressions using Yt for all k=kmin,...,kmax, 
%   where k is the # of dYt-j terms in the (A)DF regression.
%   ADF regression: dYt = rho*Yt-1 + c1*dYt-1 + c2*dYt-2 +...+ ckmax*dYt-kmax + et
%   When k=0 the regression equation becomes the DF regression (i.e. no dYt lags). 
%3. Selects the optimal lag-length k* using the following 3 methods:
%   - SIC
%   - Modified Akaike information criterion, MAIC [Ng & Perron (2001), Econometrica]
%   - Sequential-t algorithm (with alfa=10%) [Ng & Perron (1995), JASA]
%4. Estimates the DF-GLS (aka ERS) test statistic for the different 
%   optimal lag lengths, and reports these together with the Critical Values 
%   [from Cheung & Lai (1995), or Elliott, Rothenberg & Stock (1996)].
%
%INPUTS:
%- Y: The timeseries to be tested for a unit root (i.e. a Tx1 vector or timetable)
%- Trend: Defines the alternative hypothesis (Ha) of the test.
%  - 0: Yt is stationary around a mean (with no linear time trend) 
%  - 1: Yt is stationary around a linear trend
%- Kmin: The minimum # of dYt-j terms in the ADF regression; if unspecified, kmin=0
%- Kmax: The maximum # of dYt-j terms in the ADF regression; if unspecified, kmax=floor(12*((T+1)/100)^(1/4))
%- MethADF: Specifies the method for selecting the obs included in the ADF regressions.
%            0 = use SAME # of obs (T-1-kmax) across the ADF regressions for the different k's considered
%            1 = use varying # of obs (T-1-k) across the ADF regressions for the different k's considered
%- MethCVal: The method to use for retreiving the Critical Values.  
%            'CL'=Cheung & Lai (1995) 
%            'ERS'=Elliott, Rothenberg & Stock (1996)
%
%OUTPUTS: 
%- 'out': Table with the following columns:
%  - Method: The 3 optimal lag-order selection methods -> SIC, MAIC, Sequential t
%  - Opt. Lag: The optimal number of dYt lags in the ADF regression (i.e. k* -> dYt-k*) 
%  - RMSE: The RMSE convention calculated using MSE=RSS/T-1-kmax (where T is the # of obs in yt)
%  - DF-GLS: The ADF test statistic
%  - Obs: The number of observations included in the underlying ADF OLS regressions. 
%  - 1,5,10% C.Val.: The Critical Values corresponding to the 1,5 and 10% significance level. 
%                  Values for the 1% sig. lev. are always extracted from the ERS tables, 
%                  since Cheung & Lai (1995) did not provide results for the 1%.
%  - IC: Minimum SIC, Minimum MAIC, and P-value of the corresponding statistically significant 
%        coefficient of the dYt-k* term.
%
%NOTES:
%In the DF-GLS test, there are 2 possible ALTERNATIVE Hypotheses (Ha):
%  1. Ha: Yt is stationary with no linear time trend (<- case Trend=0)
%     The GLS transformed Yt on the ADF OLS regression will be: Yt* = Yt - c0
%     i.e. we de-mean the original Yt.
%  2. Ha: Yt is stationary around a linear trend (<- case Trend=1)
%     The GLS transformed Yt on the ADF OLS regression will be: Yt* = Yt - (c0+c1*t)
%     i.e. we de-trend the original Yt.

%Examples:
%out = DFGLS(y,'Trend',1,'Kmin',0,'Kmax',[],'MethADF',1,'MethCVal','CL'); %<- DEFAULTS
%out = DFGLS(y) %Equivalent to above

%Note on copyright:
%Use freely, cite responsibly. How to cite:
%Karagiannakis, H. (2024), DFGLS for MATLAB (https://github.com/ckarag/dfgls4matlab)
%Author: Haris Karagiannakis (ckarag.github.io)

%------------------------------------------------------------------------------------------%

valid = {'CL','ERS'};
check = @(z) any(validatestring(z,valid));

p = inputParser;
addRequired(p,'y',@(z) isnumeric(z) || istimetable (z));
addParameter(p,'Trend',true,@(z) isnumeric(z) || islogical(z))
addParameter(p,'Kmin',0,@isnumeric)
addParameter(p,'Kmax',[],@isnumeric)
addParameter(p,'MethADF',1,@isnumeric)
addParameter(p,'MethCVal','CL',check);
addParameter(p,'Diagnostics',true,@(z) islogical(z))

parse(p,y,varargin{:});

yt = p.Results.y;
trend = p.Results.Trend;
kmin = p.Results.Kmin;
kmax = p.Results.Kmax;
Methcval = p.Results.MethCVal;
Methadf = p.Results.MethADF;
diagnostics = p.Results.Diagnostics;

allowedset = [0,1];
if ~ismember(trend, allowedset)
  error( ['Invalid input value (trend: ' num2str(trend) '). Choose either 0 or 1.' ])
end
if ~ismember(Methadf, allowedset)
  error( ['Invalid input value (Methadf: ' num2str(Methadf) '). Choose either 0 or 1.' ])
end
if strcmpi(Methcval,'CL')
    Methcval=0;
elseif strcmpi(Methcval,'ERS')
    Methcval=1;
end

Meth.cval = Methcval;
Meth.adf = Methadf;

if istimetable(yt)
    yt = yt{:,:};
end

yt(isnan(yt)) = []; 

T=size(yt,1);
if isempty(kmin)
    kmin=0;
end
condkmax = 0;
if isempty(kmax)
    kmax=floor(12*((T+1)/100)^(1/4));
    condkmax = 1;
end
if kmax<0 || kmin<0 || kmax<kmin || mod(kmax,1)~=0 || mod(kmin,1)~=0
    error('Invalid input value. kmin & kmax should either be empty (e.g. kmax=[]), or integers with 0≤kmin≤kmax<T.' )
end
if T < 10
    error('Sample size of the inputted series cannot be less than T=10 obs.')
end
if T-kmax-1 <10 || T <= 16
    kmax = T-10-1;
    condkmax = 2;
end

[yts,~] = glsd(yt,trend);

tau = nan(kmax+1,1);
s2e = nan(kmax+1,1);
pVal = nan(kmax+1,1);
adf.Stata = nan(kmax+1,1);
adf.NgnP = nan(kmax+1,1);
Nobs.Stata = nan(kmax+1,1);
Nobs.NgnP = nan(kmax+1,1);

dyts = diffng(yts,1);
reg = nan(T,kmax+1);
reg(:,1) = lagmatrix(yts,1); 
reg(:,2:end) = lagmatrix(dyts,1:kmax);

dyts0 = dyts(kmax+2:end,:);
reg0 = reg(kmax+2:end,:);   
yt1 = reg(kmax+2:end, 1);
sumy = sum(yt1.*yt1, 'omitnan');

Tef = T-1-kmax;
k = kmin; 
while k <= kmax
    x = reg0(:,1:k+1);
    b = x\dyts0;
    e = dyts0 - x*b;
    s2e(k+1) = e'*e/Tef; 
    tau(k+1) = (b(1)^2)*sumy/s2e(k+1);
    
    
    %-----------------------------------------%
    % Calculating p-value for seq-t algorithm %
    %-----------------------------------------%
    %Calculate pValue for the LAST dYt-k term at each iteration
    dfe1 = Tef-(k+1);
    MSE = (e'*e)/dfe1;
    cov = inv(x'*x)*MSE; %COV(bols) = MSE*inv(X'X), where MSE=RSS/(T-K)
    se = sqrt(diag(cov));
    tStat = b(end)/se(end); %H0: b_true = 0 <- Saving the significance of the LAST dYt-k term
    pVal(k+1) = 2*(tcdf(-abs(tStat), dfe1)); %HA is 2-sided
    if k==0
        pVal(k+1)=NaN;
    end
    
    %-------------------------------%
    % Calculating the ADF Statistic %
    %-------------------------------%
    %Recall: The ADF test -> t-test with H0: rho=0
    
    %%%% Method 1 %%%% 
    adf.Stata(k+1) = b(1)/se(1); %<- DFGLS test statistic
    Nobs.Stata(k+1) = Tef; 
    
    %%%% Method 2 %%%%
    x = reg(k+2:end, 1:k+1); 
    b = x\dyts(k+2:end);
    e = dyts(k+2:end) - x*b;
    dfe2 = T-k-1-(k+1);
    MSE = (e'*e)/dfe2;
    cov = inv(x'*x)*MSE;
    se = sqrt(diag(cov));
    adf.NgnP(k+1) = b(1)/se(1); %<- DFGLS test statistic
    Nobs.NgnP(k+1) = T-k-1;
    
    clear x MSE cov se tStat dfe1 dfe2
    clear b e
    
    k=k+1;
end
RMSE = sqrt(s2e);

kk=(0:kmax)';

%%% MAIC %%%
mic = log(s2e)+2*(kk+tau)/Tef; %p.12 in Ng & Perron (2001)
[~, kstr] = min(mic);
kopt.MAIC = kstr-1;

%%% SIC %%%
sic = log(s2e)+log(Tef)*(kk+1)/Tef;
[~, kstr] = min(sic);
kopt.SIC = kstr-1;

%%% sequential-t algorithm %%%
alfa = 0.1; %10% significance level
optimal = find(pVal<alfa, 1,'last'); %Find the last k that is statistically significant
if isempty(optimal)
    kstr=0;
else
    kstr=optimal-1;
end
kopt.Seqt = kstr;

%--------------------%
%%% Results Tables %%%
%--------------------%
out = table();
out.Method = ["SIC", "MAIC", "Seq. t"]';
out.Opt_k = [kopt.SIC, kopt.MAIC, kopt.Seqt]';
idx_opt = [kopt.SIC+1, kopt.MAIC+1, kopt.Seqt+1]';

if Meth.adf==0
    DFGLSst = adf.Stata;
    N = Nobs.Stata;
elseif Meth.adf==1
    DFGLSst = adf.NgnP;
    N = Nobs.NgnP;
end
out.DFGLSst = DFGLSst(idx_opt);
out.Obs = N(idx_opt);
out.RMSE = RMSE(idx_opt);

%-----------------%
% Critical Values %
%-----------------%
if trend==1
    ch.tbl = "Trend";
else
    ch.tbl = "NoTrend";
end

%%%% 1. ERS (1996) Econometrica %%%%
T_ERS = repmat(T, [size(N,1) 1]); 
T_ERS(isnan(N)) = NaN;
crval = CritVal_ERSinterpol(T_ERS, ch.tbl);

%%%% 2. Cheung & Lai (1995) JBES %%%%
if Meth.cval == 0
    temp = CritVal_CnL(N, kmax, ch.tbl);
    crval.SL10 = temp.SL10;
    crval.SL5 = temp.SL5;
    clear temp
end
out.CV1 = crval.SL1(idx_opt);
out.CV5 = crval.SL5(idx_opt);
out.CV10 = crval.SL10(idx_opt);


if isempty(optimal)
    pVal_selected = NaN;
else
    pVal_selected = pVal(kopt.Seqt+1);
end
out.ICval = [sic(kopt.SIC+1), mic(kopt.MAIC+1), pVal_selected]';

if isempty(optimal) && (kmin~=0 || kmin~=1 && condkmax==0) 
    out{3, 'Opt_k'} = NaN;
end
out.Properties.VariableNames = ["Method", "Opt. Lag", "DF-GLS", "Obs", "RMSE", "1% C.Val.", "5% C.Val.", "10% C.Val.", "IC"];

if diagnostics
    if condkmax == 1
        fprintf('The maximum lag chosen (using the Schwert criterion) is %d \n',kmax);
    elseif condkmax == 2
        fprintf('The maximum lag has been adjusted (to leave at least 10 obs in the ADF regressions). Kmax was set to %d \n',kmax);
    else
        fprintf('The maximum lag chosen is %d \n',kmax);
    end
end

end


%------------------%
%%% Subfunctions %%%
%------------------%
function dx=diffng(x,k)
    if k == 0
        dx = x;
    else
        lagx = lagmatrix(x,k);
        dx = [nan(k,size(x,2)); x(1+k:end, :) - lagx(1+k:end, :)];
    end
end


function [yt,ssr] = glsd(y,trend) 
    T = size(y,1);
    if trend == 0
        z=ones(T,1);
        cbar=-7.0;
    elseif trend == 1
        z=[ones(T,1), (1:T)'];
        cbar=-13.5;
    end
    
    abar = 1+cbar/T;
    ya = zeros(T,1);
    za = zeros(T,size(z,2));
    ya(1,1) = y(1,1);
    za(1,:) = z(1,:);
    ya(2:T,1) = y(2:T,1)-abar*y(1:T-1,1);
    za(2:T,:) = z(2:T,:)-abar*z(1:T-1,:);
    bhat = za\ya;
    
    yt = y - z*bhat; 
    ssr=(ya-za*bhat)'*(ya-za*bhat);
end


function crval = CritVal_ERSinterpol(N, whichtbl)
%Interpolation of the DF-GLS Critical Values from the ERS (1996, Econometrics) table in p.825

ERS.Trend.obs = [50, 100, 200];
ERS.Trend.SL10 = [-2.89,-2.74,-2.64,-2.57]; %<- 10% significance level
ERS.Trend.SL5 = [-3.19,-3.03,-2.93,-2.89]; %<- 5% significance level
ERS.Trend.SL1 = [-3.77,-3.58,-3.46,-3.48]; %<- 1% significance level

ERS.NoTrend.obs = [25, 50, 100, 250, 500];
ERS.NoTrend.SL10 = [-1.60,-1.61,-1.61,-1.62,-1.62,-1.62]; %<- 10% significance level
ERS.NoTrend.SL5 = [-1.95,-1.95,-1.95,-1.95,-1.95,-1.95]; %<- 5% significance level
ERS.NoTrend.SL1 = [-2.66,-2.62,-2.60,-2.58,-2.58,-2.58]; %<- 1% significance level

siglvl = {'SL10','SL5','SL1'};
for sl = siglvl
    crval.(sl{:}) = nan(size(N,1),1);
    v = ERS.(whichtbl).(sl{:});
    clustobs = ERS.(whichtbl).obs;
    for i=1:size(N,1)
        if N(i) <= clustobs(1)
            crval.(sl{:})(i) = v(1);
        elseif N(i) > clustobs(end) 
            crval.(sl{:})(i) = v(end);
        elseif ~isnan(N(i)) 
            cond = ( clustobs >= N(i) );
            pos = find(diff(cond));
            crval.(sl{:})(i) = v(pos) + ((N(i) - clustobs(pos))/clustobs(pos))*(v(pos+1) - v(pos));
        else 
            crval.(sl{:})(i) = NaN;
        end
    end
end
end


function crval = CritVal_CnL(N, kmax, whichtbl)
%Critical values for the DF-GLS test computed using the response function
%of Cheung & Lai (1995) OBES.

iN = 1./N;
kk=(0:kmax)';
kbo = kk./N;

%Cheung and Lai OBES 1995 Table 1, cols 3-4
CnL.Trend.c10 = [-2.550, -20.166, 155.215, 1.133, 9.808, -20.313]; %<- 10% significance level
CnL.Trend.c5 = [-2.838, -20.328, 124.191, 1.267, 10.530, -24.600]; %<- 5% significance level
%Cheung and Lai OBES 1995 Table 1, cols 1-2
CnL.NoTrend.c10 = [-1.624, -19.888, 155.231, 0.709, 5.480, -16.055]; %<- 10% significance level
CnL.NoTrend.c5 = [-1.948, -17.839, 104.086, 0.802, 5.558, -18.332]; %<- 5% significance level

v = CnL.(whichtbl).c5;
crval.SL5 = v(1) + v(2)*iN + v(3)*iN.^2 + v(4)*kbo + v(5)*kbo.^2 + v(6)*kbo.^3;
v = CnL.(whichtbl).c10;
crval.SL10 = v(1) + v(2)*iN + v(3)*iN.^2 + v(4)*kbo + v(5)*kbo.^2 + v(6)*kbo.^3;
end