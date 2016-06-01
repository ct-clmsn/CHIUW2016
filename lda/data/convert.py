#!/usr/bin/env python
'''
        D           1x1                          8  double
        N           1x1                          8  double
        W           1x1                          8  double
        d      410595x1                    3284760  double
        w      410595x1                    3284760  double
        word     6906x1                      48561  cell
'''
from scipy.io import *
from scipy.sparse import lil_matrix
KOS_TRAINFILE='docword.kos.train.mat'
NIPS_TRAINFILE='docword.nips.train.mat'

for f in (KOS_TRAINFILE, NIPS_TRAINFILE):
   m = loadmat(f)
   for k in ['D', 'N', 'W', 'd', 'w', 'word']:
      if k == 'word':
         with open('%s.%s' % (f,k), 'w') as fd:
            for word in m[k]:
               print word[0][0]
               fd.write('%s\n' % (word[0][0].strip(),)) 
      else:          
         A = lil_matrix(m[k], dtype=float)
         mmwrite('%s.%s' % (f,k), A)
         

