#include "mex.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * plhs[0] = *z
 * plhs[1] = **wp
 * plhs[2] = **dp
 * 
 * prhs[0] = *z
 * prhs[1] = **wp
 * prhs[2] = **dp
 * prhs[3] = *ztot
 * prhs[4] = *w
 * prhs[5] = *d
 *
 */

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
  int N;
  int W;
  int D;
  int T;

  double alpha = 0.1;
  double beta = 0.01;
  double wbeta;

  double *w;
  double *d;
  double *z;
  double *wp;
  double *dp;
  double *ztot;

  int i, t, wi, di;
  double totprob, maxprob, currprob;
  double *probs;

  double *z_;
  double *wp_;
  double *dp_;
  
  N = mxGetM(prhs[0]) * mxGetN(prhs[0]);
  W = mxGetM(prhs[1]);
  T = mxGetN(prhs[1]);
  D = mxGetM(prhs[2]);

  z    = mxGetPr(prhs[0]);
  wp   = mxGetPr(prhs[1]);
  dp   = mxGetPr(prhs[2]);
  ztot = mxGetPr(prhs[3]);
  w    = mxGetPr(prhs[4]);
  d    = mxGetPr(prhs[5]);
  
  probs = mxMalloc(T * sizeof(double));
  wbeta = W*beta;
  
  plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(W,T,mxREAL);
  plhs[2] = mxCreateDoubleMatrix(D,T,mxREAL);
  
  z_  = mxGetPr(plhs[0]);
  wp_ = mxGetPr(plhs[1]);
  dp_ = mxGetPr(plhs[2]);
  
  /* copy input arrays to output arrays */
  for (i = 0; i < N; i++) {
    w[i]--;
    d[i]--;
    z_[i] = z[i]-1;
  }
  for (i = 0; i < W*T; i++) wp_[i] = wp[i];
  for (i = 0; i < D*T; i++) dp_[i] = dp[i];

  
  /******************************************************/
  for (i = 0; i < N; i++) {

    wi = (int)(w[i]);
    di = (int)(d[i]);
    
    t = z_[i];
    ztot[t]--;     
    wp_[t*W + wi]--;
    dp_[t*D + di]--;

    totprob = 0;
    for (t = 0; t < T; t++) {
//printf("( (%d, %d), (%f, %f, %f) )\n", wi, di, wp_[t*W + wi], dp_[t*D + di], ztot[t]);
      probs[t] = (wp_[t*W + wi] + beta) * (dp_[t*D + di] + alpha) / (ztot[t] + wbeta);
      totprob += probs[t];
    }
    
    maxprob  = totprob * drand48();
    currprob = probs[0];
    t = 0;
    while (maxprob > currprob) {
      t++;
      currprob += probs[t];
    }
   
    z_[i] = t;
    ztot[t]++;     
    wp_[t*W + wi]++;
    dp_[t*D + di]++;

  }

  for (i = 0; i < N; i++) {
    z_[i]++;
    w[i]++;
    d[i]++;
  }

  mxFree(probs);

}
