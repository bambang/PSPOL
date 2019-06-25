'''
The algorithm in PSPOLFIL is based on the article in the reference section. If the input data set is in the scattering matrix format, it is converted to the covariance matrix format. The total power image is then created as in the PSTOTPOW program. Every pixel P(i,j) of the total power image represents the sum of intensities of all polarizations in the input data set.

The total power image is then averaged with a 3x3 window as follows:

                   1    1     1
        Pav(i,j) = - * Sum ( Sum ( P(i'+i,j'+j) ) )
                   9  i'=-1 j'=-1
      
Several working parameters are derived as follows:

        L = NUMLK
        N = WINSIZE
        m = (N - 3) / 2
        n = (N + 1) / 2
        N2 = (N * (N + 1)) / 2
      
PSPOLFIL then loops over all lines and pixels of the input data set. The processing at every pixel proceeds as follows:

At the current pixel of the averaged total power image, (i,j), compute edge strengths, wd, in four directions (d=1,2,3,4) as follows:
                   1     1
            wd = | Sum ( Sum ( Wd(i',j') * Pav(m*i'+i,m*j'+j) ) ) |
                   i'=-1 j'=-1
          
The four edge detection windows, Wd, are defined as follows:

                 (-1  0  1)        (-1 -1 -1)        (-1 -1  0)        ( 0 -1 -1)
            W1 = (-1  0  1)   W2 = ( 0  0  0)   W3 = (-1  0  1)   W4 = ( 1  0 -1)
                 (-1  0  1)        ( 1  1  1)        ( 0  1  1)        ( 1  1  0)
          
Find the direction, s, that yields the strongest edge. It is given by the maximum of the four edge strengths, as follows:
            ws = max ( w1, w2, w3, w4 )
          
Use the strongest edge direction, s, to select one of the eight averaging windows Fk (k=1,...,8) that are used to filter the central pixel.
All elements of the NxN windows Fk are either 0 or 1. A pixel set to 1 selects the corresponding image pixel to contribute to filtering of the central pixel. A pixel set to 0 eliminates the corresponding image pixels from filtering. Four windows have 1-valued pixels in the left, right, upper half, or lower half, including the central row or column. The remaining four windows have 1-valued pixels in the upper-left, upper-right, lower-left, or lower-right quadrant, including the right or left diagonal.

The two Fk windows that are aligned with the strongest edge, ws, are examined. The window with its avearge power closest to the Pav of the central pixel is selected and is represented as F.

Estimate the mean, mu, and variance, nu, of the total power in the filtering window, F, as follows:
                  1     n     n
            mu = --- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j) ) )
                  N2  i'=-n j'=-n
                  1     n     n                                      N2
            nu = ---- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j)^2 ) ) -  ---- * mu^2
                 N2-1  i'=-n j'=-n                                   N2-1
          
Compute the filter weight, b, as follows:
                     L*nu - mu^2
            b = max( -----------, 0 )
                     (L+1) * nu
          
Loop over all elements (channels) of the input polarimetric matrix. Filter the current polarimetric element as follows:
                                   1-b    n     n
            Vf(i,j) = b * V(i,j) + --- * Sum ( Sum ( F(i',j') * V(i'+i,j'+j) ) )
                                    N2  i'=-n j'=-n
          
V(i,j) represents the original value of the polarimetric matrix element at the current pixel. Vf(i,j) represents the filtered value of the same element. It is stored in the output channel as a floating point pixel value.
'''
from scipy.signal import convolve2d
import numpy as np
import gdal

def generate_pav(Ptot, n=3):
    filtr = np.ones((n,n)) / n**2
    output = convolve2d(Ptot, filtr, boundary='symm', mode='same')
    return output
    
def Wd(d, n=3):
    if d==0:
       Wd = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    elif d==1:
        Wd = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
    elif d==2:
        Wd = np.flip(np.ones((n,n)) + np.triu( np.repeat(-2,n)) + np.identity(n), 1)
    elif d==3:
        Wd = np.ones((n,n)) + np.triu( np.repeat(-2,n)) + np.identity(n)
    return Wd
    

def Fk(k, n):
    if k==0:
        Fk = np.ones((n,n))
        Fk[:, 0:(n//2)] = 0
    elif k==1:
        Fk = np.triu(np.repeat(1, n))
    elif k==2:
        Fk = np.ones((n,n))
        Fk[(n//2 + 1):n, :] = 0
    elif k==3:
        Fk = np.flip(np.triu(np.repeat(1, n)), 1)
    elif k==4:
        Fk = np.ones((n,n))
        Fk[:,(n//2 + 1):n] = 0     
    elif k==5:
        Fk = np.tril(np.repeat(1, n))
    elif k==6:
        Fk = np.ones((n,n))
        Fk[0:(n // 2) , :] = 0   
    elif k==7:
        Fk = np.flip(np.tril(np.repeat(1, n)), 1)
    return(Fk)

def fk_from_wd(d):
    lookup = {0: [0, 4],
              1: [2, 6],
              2: [3, 7],
              3: [1, 5]}
    return(lookup[d])
    
def check_fk_avg(P, i, j, n, N2, Fk):
    avg = 0
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            avg += Fk[ip+n,jp+n] * P[i + ip, j + jp]
    avg /= N2
    return avg

def choose_fk(Pav, i, j, n, N2, F1, F2):
    '''The two Fk windows that are aligned with the strongest edge, ws, are examined. 
    The window with its avearge power closest to the Pav of the central pixel is selected and is represented as F.
    '''
    val = Pav[i, j] 
    diff1 = np.abs(check_fk_avg(Pav, i, j, n, N2, F1) - val)
    diff2 = np.abs(check_fk_avg(Pav, i, j, n, N2, F2) - val)
    
    if diff1 < diff2:
        return F1
    else:
        return F2
    

        
def wd(Pav, i, j, Wd, m):
    '''At the current pixel of the averaged total power image, (i,j), compute edge strengths, wd, in four directions (d=1,2,3,4) as follows:
                   1     1
            wd = | Sum ( Sum ( Wd(i',j') * Pav(m*i'+i,m*j'+j) ) ) |
                   i'=-1 j'=-1
    '''
    wd = 0
    for ip in [-1,0,1]:
        for jp in [-1,0,1]:
            wd += Pav[ip*m + i, jp*m + i] * Wd[ip+1, jp+1]
    return np.abs(wd)

def ws(Pav, i, j, m):
    '''
    Find the direction, s, that yields the strongest edge. It is given by the maximum of the four edge strengths, as follows:
            ws = max ( w1, w2, w3, w4 )
    Returns
    -------
    int
        index of strongest edge detection matrix
    '''
    greatest = 0
    ix = 0
    
    for d in range(4):
        current = wd(Pav, i, j, WD[d], m)
        if current > greatest:
            greatest = current
            ix = d
    
    return(ix)
   
def mu(P, i, j, F, n, N2):
    '''
    Mean of total power within F-window
          1     n     n
    mu = --- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j) ) )
          N2  i'=-n j'=-n
        '''
    sigma = 0
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            sigma += F[ip+n, jp+n] * P[ip + i, jp + j]
    return sigma / N2
    

def nu(P, i, j, F, n, N2, mu):
    '''
           1     n     n                                      N2
    nu = ---- * Sum ( Sum ( F(i',j') * P(i'+i,j'+j)^2 ) ) -  ---- * mu^2
         N2-1  i'=-n j'=-n                                   N2-1
    '''
    sigma = 0
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            sigma += F[ip+n, jp+n] * np.power(P[ip + i, jp + j], 2)
            
    result = (1 / (N2 - 1)) * sigma - (N2 * mu**2) / (N2 - 1)
    return result

def weight(L, nu, mu):
    '''
Compute the filter weight, b, as follows:
                     L*nu - mu^2
            b = max( -----------, 0 )
                     (L+1) * nu
    '''    
    result = max(((L * nu - mu**2) / ((L + 1) * nu)), 0)
    return result

    
def Vf(V, i, j, n, N2, b, F):
    '''
Loop over all elements (channels) of the input polarimetric matrix. Filter the current polarimetric element as follows:
                           1-b    n     n
    Vf(i,j) = b * V(i,j) + --- * Sum ( Sum ( F(i',j') * V(i'+i,j'+j) ) )
                           N2  i'=-n j'=-n  
    '''
    sigma = 0
    for ip in range(-n, n+1):
        for jp in range(-n, n+1):
            sigma += F[ip+n, jp+n] * V[ip + i, jp + j]
    result = b * V[i, j] + ((1 - b) / N2) * sigma
    return result
            
def PSPOLFIL(img, P, NUMLK, WINSIZE):
    ''' 
    img : array-like
        array with shape (z,y,x) where the channel varies along z
    '''
    L = NUMLK
    N = WINSIZE
    m = int((N - 3) / 2)   # offset to calculate mean value
    n = int((N - 1) / 2)   # length of filter kernel on either side of centre
    N2 = (N * (N + 1)) / 2 # number of '1' elements in in F kernel
    
    WD = [Wd(i) for i in range(4)]
    FK = [Fk(i, N) for i in range(8)]
    
    output = np.empty_like(img)

    P_pad = np.pad(P, n, 'symmetric')
    Pav = generate_pav(P_pad)
    
    img_pad = np.pad(img, ((0,0),(2,2),(2,2)), 'symmetric')
    pdone=0
    
    img_x = img.shape[1]
    img_y = img.shape[2]
    
    for ii in range(img_x):
        i = ii + n
        
        for jj in range(img_y):
            j = jj + n
            
            strongest_edge_d = ws(Pav=Pav, i=i, j=j, m=m )
            f1, f2 = fk_from_wd(d=strongest_edge_d)
            F = choose_fk(P_pad, i, j, n, N2, F1=FK[f1], F2=FK[f2])
            
            Mu = mu(P_pad, i, j, F, n, N2)
            Nu = nu(P_pad, i, j, F, n, N2, Mu)
            
            b = weight(L=L, nu=Nu, mu=Mu)
            for channel in range(img.shape[0]):
                output[channel, ii, jj] = Vf(img_pad[channel, :, :], i, j, n, N2, b, F) 
                
            if 100 * (i*j) / (img_x*img_y) > pdone:
                print(pdone)
                pdone +=1
                
    return(output)
    
    
    
"""

HH = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HH_complex.tif") 
hh = HH.ReadAsArray(xoff=5000, yoff=5000, xsize=250, ysize=250)   
hh = hh[0] + 1j*hh[1]
HV = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HV_complex.tif")       
hv = HV.ReadAsArray(xoff=5000, yoff=5000, xsize=250, ysize=250)    
hv = hv[0] + 1j*hv[1]
    
sar =   np.stack( [hh, hv], axis=2 )
sar = np.moveaxis(sar, 2,0)
#sar = sar[:,175:225,175:225]
pow = np.absolute(sar/4.370677e+03)**2
totpow = np.sum(pow, axis=0)   
pav = generate_pav(totpow, n=3)   
    
#out2  = PSPOLFIL(img=np.absolute(sar), P=totpow, NUMLK=1, WINSIZE=5)    
    
FK = [Fk(i, 7) for i in range(8)]    
    
import matplotlib.pyplot as plt
def rescale(img):
    res = np.array(img, dtype=np.float32)
    p = np.percentile(img, (2,98))
    res = img / (p[1] / 255)
    res[np.greater(res, 255)] = 255
    return(np.array(res, dtype='int16'))


fig, (ax_orig, ax_mag, ax_ang, p4) = plt.subplots(4, 1, figsize=(6, 15))
ax_orig.imshow(rescale(pow[0,:,:]), cmap='gray')
ax_orig.set_title('HH Power')
ax_orig.set_axis_off()
ax_mag.imshow(rescale(pow[1,:,:]), cmap='gray')
ax_mag.set_title('HV Power')
ax_mag.set_axis_off()
ax_ang.imshow(rescale(totpow), cmap='gray') # hsv is cyclic, like angles
ax_ang.set_title('Total Power')
ax_ang.set_axis_off()
p4.imshow(rescale(np.absolute(out2[0,:,:])), cmap='gray') # hsv is cyclic, like angles
p4.set_title('filtered HH')
p4.set_axis_off()
fig.show()
    
    
rgb = np.concatenate((rescale(out2[[0],:,:]), rescale(out2[[1],:,:]), rescale(out2[[0],:,:])), axis=0) 
rgb = np.moveaxis(rgb, 0, 2)    
    
fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
hh.imshow(rescale(np.absolute(sar[0,:,:])), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
p1.imshow(rgb) # hsv is cyclic, like angles
p1.set_title('filtered HH')
p1.set_axis_off()
fig.show()
    
    
    
    
fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
hh.imshow(rescale(totpow), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
p1.imshow(rescale(pav), cmap='gray') # hsv is cyclic, like angles
p1.set_title('filtered HH')
p1.set_axis_off()
fig.show()


fig, hh= plt.subplots(1, 1, figsize=(15, 15))
hh.imshow(rescale(np.absolute(sar[1,175:225,175:225])), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
fig.show()

fig, hh= plt.subplots(1, 1, figsize=(15, 15))
hh.imshow(rescale(totpow), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
fig.show()   

ex = np.array([99, 105,124,138,128,34,62,
               105,91,140,98,114,63,31,
               107,94,128,138,96,61,82,
               137,129,136,105,100,55,85,
               144,145,113,132,119,39,50,
               102,97,102,110,103,34,53,
               107,146,115,123,101,76,56]).reshape((7,7))   
    
exav = generate_pav(ex, 3)    


L = NUMLK
N = WINSIZE
m = int((N - 3) / 2)
n = int((N - 1) / 2)
N2 = (N * (N + 1)) / 2

def filter_i_j(P, Pav, i, j, L, n, N2): 
    strongest_edge_d = ws(Pav, i, j, 2)
    f1, f2 = fk_from_wd(d=strongest_edge_d)
    F = choose_fk(P, i, j, n, N2, F1=FK[f1], F2=FK[f2])
    
    Mu = mu(P, i, j, F, n, N2)
    Nu = nu(P, i, j, F, n, N2, Mu)
    b = weight(L, Nu, Mu)
    
    return(b, F)
        
filter_i_j(totpow, pav, 10, 11, 1,  3, 15)
"""