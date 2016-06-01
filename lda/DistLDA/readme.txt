Distributed LDA Package (Version 1.0, last modified 11/12/10).
Arthur Asuncion (asuncion '@' uci.edu).

This folder contains a MATLAB simulation of distributed sampling for Latent 
Dirichlet Allocation. First, one needs to compile the MEX file in MATLAB:

>> mex gibbsMex.c

Two sample text data sets are included, KOS and NIPS.  To run the sampler
on NIPS using T=10 topics and P=4 processors, one can simply call the following:

>> distributedLDAsampler('n',10,4,1);
