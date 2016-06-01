% plhs[0] = *z
% plhs[1] = **wp
% plhs[2] = **dp

% prhs[0] = *z
% prhs[1] = **wp
% prhs[2] = **dp
% prhs[3] = *ztot
% prhs[4] = *w
% prhs[5] = *d

function [z_, wp_, dp_] = gibbs(z, wp, dp, ztot, w, d)
  alpha = 0.1;
  beta = 0.01;
  wbeta = 0.0;

  totprob = 0.0;
  maxprob = 0.0;
  currprob = 0.0;
  
  N = size(z,1) * size(z,2); % mxGetM(prhs[0]) * mxGetN(prhs[0]);
  W = size(wp,1); % mxGetM(prhs[1]);
  T = size(wp,2); % mxGetN(prhs[1]);
  D = size(dp,1); % mxGetM(prhs[2]);

  % z    = mxGetPr(prhs[0]);
  % wp   = mxGetPr(prhs[1]);
  % dp   = mxGetPr(prhs[2]);
  % ztot = mxGetPr(prhs[3]);
  % w    = mxGetPr(prhs[4]);
  % d    = mxGetPr(prhs[5]);
  
  probs = zeros(T);
  wbeta = W*beta;
  
  % plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
  % plhs[1] = mxCreateDoubleMatrix(W,T,mxREAL);
  % plhs[2] = mxCreateDoubleMatrix(D,T,mxREAL);
  
  z_  = z; % mxGetPr(plhs[0]);
  wp_ = wp; % mxGetPr(plhs[1]);
  dp_ = dp; % mxGetPr(plhs[2]);
  
  % copy input arrays to output arrays
  w-=1;
  d-=1;
  z_ = z - 1;
  %randvals = floor(rand(N) * (mx - mn + 1.0)) + mn;
  mn = 0.0;
  mx = 1.0;

  for i = 1:N %i = 0; i < N; i++) {
    wi = round(w(i))+1;
    di = round(d(i))+1;
    
    t = z_(i)+1;
    ztot(t)--;     
    wp_(wi,t)--;
    dp_(di,t)--;

    totprob = 0;
    probs = (wp_(wi,:) + beta) .* (dp_(di,:)+alpha) / (ztot + wbeta);
    
    totprob = sum(probs);
    maxprob  = totprob * ((rand(1) * (mx - mn + 1.0)) + mn); %drand48(); % normalize rand(1) between 0.0 and 1.0
    currprob = probs(1);
    t = 1;
    while (maxprob > currprob)
      t++;
      currprob += probs(t);
    end
   
    z_(i) = t;
    ztot(t)++;     
    wp_(wi,:)++;
    dp_(di,:)++;

  end  

  z_+=1;
  w+=1;
  d+=1;

end

