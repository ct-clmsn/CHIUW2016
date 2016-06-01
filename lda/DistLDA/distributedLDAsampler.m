function [] = distributedLDAsampler(dataset,T,P,seed)
% function [] = distributedLDAsampler(dataset,T,P,seed)
%
% Runs approximate distributed collapsed Gibbs sampling for LDA.
% This is a Matlab simulation of the distributed sampler.
%
% INPUTS:
%   dataset: 'k' for KOS data and 'n' for NIPS data
%   T: number of topics
%   P: Number of processors
%   seed: random seed
%
% Note: For simplicity, we assume that P evenly divides
% the number of documents in the data set.

% parameters
%dataset
%T
%P
ITER  = 500;
beta  = 0.01;
alpha = 0.1;
rand('state',sum(100*clock))

% read corpus
if (dataset=='k')
  load ../data/docword.kos.train.mat
elseif (dataset=='n')
  load ../data/docword.nips.train.mat
end

% random initial assignment of topic assignments
z  = floor(T*rand(N,1)) + 1;
% W x T and D x T topic count matrices
wp = zeros(W,T);
dp = zeros(D,T);
for n = 1:N
  wp(w(n),z(n)) = wp(w(n),z(n)) + 1;
  dp(d(n),z(n)) = dp(d(n),z(n)) + 1;
end
ztot = sum(wp,1);
ztotchk = sum(dp,1);

% split over P procs
Dp = D/P;
Nstart = 1;

for p=1:P
  
  Dstart = (p-1)*Dp + 1;
  Dend   =     p*Dp;
  
  Nend   = full(sum(sum(dp(1:Dend,:))));

  data(p).w  =  w( Nstart:Nend );
  data(p).d  =  d( Nstart:Nend ) - (Dstart - 1); % local numbering
  data(p).z  =  z( Nstart:Nend );
  data(p).dp = dp( Dstart:Dend, : );
  
  Nstart = Nend + 1;
  
end

tic;
% iterate through distributed sampling
for iter = 1:ITER
    
  wp0 = wp;
  ztot0 = sum(wp0);
  dwp = 0*wp;
  for p=1:P
    %[data(p).z, wp, data(p).dp] = gibbsMex(data(p).z, wp0, data(p).dp, ztot0, data(p).w, data(p).d);
    [data(p).z, wp, data(p).dp] = gibbs(data(p).z, wp0, data(p).dp, ztot0, data(p).w, data(p).d);
    dwp = dwp + (wp - wp0);
  end
  wp = wp0 + dwp;
  
  % print topics
  %if (mod(iter,1)==0)
  %  for t=1:T
  %    [xsort,isort] = sort(-wp(:,t));
  %    fprintf('[t%d] (%.3f) ', t, ztot0(t)/N);
  %    for i=1:min(8,W)
  %      fprintf('%s ', word{isort(i)});
  %    end
  %    fprintf('\n');
  %  end
  %end

end
toc;

