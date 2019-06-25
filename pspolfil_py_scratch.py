    

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
    
    
   