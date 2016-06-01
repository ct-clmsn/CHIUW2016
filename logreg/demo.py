from sklearn.preprocessing import scale
import numpy as np

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from sys import argv
from logreg import *

X0, y0 = mmread(argv[1]), mmread(argv[2])

X0 = X0.todok()
y0 = y0.todok() 

from scipy.sparse import csr_matrix 
X = csr_matrix(scale(X0.todense()[:,0:10]), dtype=float)
y = csr_matrix(y0.todense(), dtype=float)

# shuffle for SGD
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

Xr, yr = X[indices,:].astype(float), y[0,indices].todense().astype(float)

from timeit import default_timer as timer
start = timer()

w = train(Xr, yr, alpha=0.3)

end = timer()

print(end - start) 

