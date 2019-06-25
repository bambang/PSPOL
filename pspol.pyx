cimport cython
from libc.math cimport fmax, abs
import numpy as np
from scipy.signal import convolve2d


# cython: profile=True


cdef double[:,:] generate_pav(double[:,:] Ptot, int n=3):
    filtr = np.ones((n,n)) / n**2
    output = convolve2d(Ptot, filtr, boundary='symm', mode='same')
    return output
    
cdef int[:,:] Wd(int d, int n=3):
    cdef int[:,:] Wd
    if d==0:
        Wd = np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype='int32')
    elif d==1:
        Wd = np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype='int32')
    elif d==2:
        Wd = np.array(np.flip(np.ones((n,n)) + np.triu( np.repeat(-2,n)) + np.identity(n), 1), dtype='int32')
    elif d==3:
        Wd = np.array(np.ones((n,n)) + np.triu( np.repeat(-2,n)) + np.identity(n), dtype='int32')
    return Wd
    

cdef int[:,:] Fk(int k, int n):
    ''' The index-array mapping is different than the python implementation
    which uses the original indices based on Lee(1999).  This is done so that '''
    cdef int[:,:] Wd
    if k==0:
        Fk = np.ones((n,n), dtype='int32')
        Fk[:, 0:(n//2)] = 0
    elif k==1:
        Fk = np.array(np.triu(np.repeat(1, n)), dtype='int32')
    elif k==2:
        Fk = np.ones((n,n), dtype='int32')
        Fk[(n//2 + 1):n, :] = 0
    elif k==3:
        Fk = np.array(np.flip(np.triu(np.repeat(1, n)), 1), dtype='int32')
    elif k==4:
        Fk = np.ones((n,n), dtype='int32')
        Fk[:,(n//2 + 1):n] = 0     
    elif k==5:
        Fk = np.array(np.tril(np.repeat(1, n)),  dtype='int32')
    elif k==6:
        Fk = np.ones((n,n), dtype='int32')
        Fk[0:(n // 2) , :] = 0   
    elif k==7:
        Fk = np.array(np.flip(np.tril(np.repeat(1, n)), 1), dtype='int32')
    return(Fk)


   
cdef int[:] fk_from_wd(int d):
    cdef dict lookup = {0: [0, 4],
                        1: [2, 6],
                        2: [3, 7],
                        3: [1, 5]}
    cdef int[2] result = lookup[d]
    return(result)
    
@cython.profile(True)    
cdef double check_fk_avg(double[:,:] P, int i, int j, int n, double N2, int[:,:] Fk):
    cdef double avg = 0
    cdef int ip, jp
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            avg += Fk[ip+n,jp+n] * P[i + ip, j + jp]
    avg /= N2
    return avg

cdef choose_fk(double[:,:]Pav, int i, int j, int n, double N2, int[:,:] F1, int[:,:] F2):
    '''The two Fk windows that are aligned with the strongest edge, ws, are examined. 
    The window with its avearge power closest to the Pav of the central pixel is selected and is represented as F.
    '''
    cdef double val = Pav[i, j] 
    cdef double diff1 = abs(check_fk_avg(Pav, i, j, n, N2, F1) - val)
    cdef double diff2 = abs(check_fk_avg(Pav, i, j, n, N2, F2) - val)
    
    if diff1 < diff2:
        return F1
    else:
        return F2
        
@cython.profile(True)                
cdef double wd(double[:,:] Pav, int i, int j, int[:,:] Wd, int m):
    '''At the current pixel of the averaged total power image, (i,j), compute edge strengths, wd, in four directions (d=1,2,3,4) as follows:
                   1     1
            wd = | Sum ( Sum ( Wd(i',j') * Pav(m*i'+i,m*j'+j) ) ) |
                   i'=-1 j'=-1
    '''
    cdef double wd = 0
    cdef int ip, jp
    
    for ip in range(-1,2):
        for jp in range(-1,2):
            wd += Pav[ip*m + i, jp*m + i] * Wd[ip+1, jp+1]
    return abs(wd)
    
@cython.profile(True)    
cdef int ws(double[:,:] Pav, int i, int j, int m, int[:,:,:] WD):
    '''
    Find the direction, s, that yields the strongest edge. It is given by the maximum of the four edge strengths, as follows:
            ws = max ( w1, w2, w3, w4 )
    Returns
    -------
    int
        index of strongest edge detection matrix
    '''
    cdef double greatest = 0
    cdef int ix = 0
    cdef int d
    cdef double current
    
    for d in range(4):
        current = wd(Pav, i, j, WD[d,:,:], m)
        if current > greatest:
            greatest = current
            ix = d
    
    return(ix)

@cython.profile(True)
@cython.cdivision(True)
cdef double mu(double[:,:] P, int i, int j, int[:,:] F, int n, double N2):
    '''
    Mean of total power within F-window
          1     n     n
    mu = --- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j) ) )
          N2  i'=-n j'=-n
        '''
    cdef double sigma = 0
    cdef Py_ssize_t ip, jp
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            sigma += F[ip+n, jp+n] * P[ip + i, jp + j]
    sigma /= N2
    return sigma 

@cython.profile(True)
@cython.cdivision(True)
cdef double nu(double[:,:] P, int i, int j, int[:,:] F, int n, double N2, double mu):
    '''
           1     n     n                                      N2
    nu = ---- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j)^2 ) ) -  ---- * mu^2
         N2-1  i'=-n j'=-n                                   N2-1
    '''
    cdef double sigma = 0
    cdef Py_ssize_t ip, jp
    cdef double array_val
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            array_val = P[ip + i, jp + j]
            sigma += F[ip+n, jp+n] * array_val*array_val
  
    cdef double result = (1 / (N2 - 1)) * sigma - (N2 * mu*mu) / (N2 - 1)
    return result
    
@cython.profile(True)    
cdef double weight(int L, double nu, double mu):
    '''
Compute the filter weight, b, as follows:
                     L*nu - mu^2
            b = max( -----------, 0 )
                     (L+1) * nu
    ''' 
    cdef double result = fmax(((L * nu - mu*mu) / ((L + 1) * nu)), 0)
    return result

@cython.profile(True)
cdef double Vf(double[:,:] V, int i, int j, int n, double N2, double b, int[:,:] F):
    '''
Loop over all elements (channels) of the input polarimetric matrix. Filter the current polarimetric element as follows:
                           1-b    n     n
    Vf(i,j) = b * V(i,j) + --- * Sum ( Sum ( F(i',j') * V(i'+i,j'+j) ) )
                           N2  i'=-n j'=-n  
    '''
    cdef double sigma = 0
    cdef Py_ssize_t ip, jp
    
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            sigma += F[ip+n, jp+n] * V[ip + i, jp + j]
    cdef double result = b * V[i, j] + ((1 - b) / N2) * sigma
    return result
    
  
cpdef double[:,:,:] PSPOLFIL(double[:,:,:] img, double[:,:] P, int NUMLK, int WINSIZE):
    ''' 
    img : array-like
        array with shape (z,y,x) where the channel varies along z
    '''
    cdef int L = NUMLK
    cdef int N = WINSIZE
    cdef int m = (N - 3) / 2   # offset to calculate mean value
    cdef int n = (N - 1) / 2   # length of filter kernel on either side of centre
    cdef double N2 = (N * (N + 1)) / 2 # number of '1' elements in in F kernel
    
    cdef int ii, i, jj, j, f1, f2, strongest_edge_d, channel
    cdef double Mu, Nu, b
    
    #WD = [Wd(d) for d in range(4)]
    cdef int[:,:,:] WD  = np.empty((4,3,3), dtype='int32')
    for i in range(4):
        WD[i,:,:] = Wd(i) 
        
    #FK = [Fk(k, N) for k in range(8)]
    cdef int[:,:,:] FK = np.empty((8,N,N), dtype='int32')
    for i in range(8):
        FK[i,:,:] = Fk(i, N)
        
    cdef double[:,:,:] output = np.empty_like(img)

    cdef double[:,:] P_pad = np.pad(P, n, 'symmetric')
    cdef double[:,:] Pav = generate_pav(P_pad)
    
    cdef double[:,:,:] img_pad = np.pad(img, ((0,0),(2,2),(2,2)), 'symmetric')
    cdef int pdone = 0
    
    cdef Py_ssize_t img_x = img.shape[1]
    cdef Py_ssize_t img_y = img.shape[2]

    for ii in range(img_x):
        i = ii + n
        
        for jj in range(img_y):
            j = jj + n
            
            strongest_edge_d = ws(Pav=Pav, i=i, j=j, m=m, WD=WD)
            f1, f2 = fk_from_wd(d=strongest_edge_d)
            F = choose_fk(P_pad, i, j, n, N2, F1=FK[f1,:,:], F2=FK[f2,:,:])
            
            Mu = mu(P_pad, i, j, F, n, N2)
            Nu = nu(P_pad, i, j, F, n, N2, Mu)
            
            b = weight(L=L, nu=Nu, mu=Mu)
            
            for channel in range(img.shape[0]):
                output[channel, ii, jj] = Vf(V=img_pad[channel, :, :], i=i, j=j, n=n, N2=N2, b=b, F=F) 
                
            if 100 * (i*j) / (img_x*img_y) > pdone:
                print(pdone)
                pdone +=1
                
    return(output)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    