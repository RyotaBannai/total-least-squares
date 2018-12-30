import numpy as np
import numpy.linalg as la

def tls(X,y):
    if len(X.shape) is 1:
        n = 1
        X = X.reshape(len(X),1)
    else:
        n = np.array(X).shape[1] # the number of variable of X
    
    Z = np.vstack((X.T,y)).T
    U, s, Vt = la.svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy  / Vyy # total least squares soln
    
    Xtyt = - Z.dot(V[:,n:]).dot(V[:,n:].T)
    Xt = Xtyt[:,:n] # X error
    y_tls = (X+Xt).dot(a_tls)

    fro_norm = la.norm(Xtyt, 'fro')#Frobenius norm
    
    return y_tls, X + Xt, a_tls, fro_norm
