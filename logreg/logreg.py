from numpy import exp
import numpy as np

def predict(w, x):
    x = np.asarray(x[0,:].todense())
    wTx = np.dot(w, np.hstack((x, [[1.0,]])).T)
    return 1. / (1. + exp(- wTx ))

def update(w, x, y, alpha):
    p = predict(w, x)
    Z = np.column_stack((np.asarray(x.todense()), np.array([1.0,])))
    w = w - (alpha * (p - y) * Z)
    
def train(X, y, alpha, w=None):
    n, m = X.shape
    if w is None:
        w = np.zeros(m + 1)
    for i in xrange(n):
        update(w, X[i,:], y[0,i], alpha)
    return w

from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
import numpy as np
if __name__ == "__main__":
   data = load_iris()
   X0, y0 = data.data, data.target
   y = y0[y0 != 0]
   y[y == 1] = 0
   y[y == 2] = 1
   X = X0[y0 != 0, :]

   from scipy.sparse import csr_matrix 
   X = csr_matrix(scale(X[:, [2, 3]]), dtype=float)
   y = csr_matrix(y, dtype=float)

   from scipy.io import mmwrite
   mmwrite("scaled_data", X)
   mmwrite("scaled_lbls", y)

   # shuffle for SGD
   indices = np.arange(X.shape[0])
   np.random.shuffle(indices)
   Xr, yr = X[indices,:], y[0,indices]

   # train
   w = train(Xr.todense(), np.asarray(yr[0,:].todense(), dtype=float), alpha=0.3)
   print 'weight:', w

   # predict
   #p = [predict(w, x) for x in Xr]
   #print np.mean(yr == map(round, p))

