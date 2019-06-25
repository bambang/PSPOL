from DUAP.PSPOL.pspol import PSPOLFIL as ps_c
from DUAP.PSPOL.pspol import PSPOLFIL_complex as ps_c_com
from DUAP.PSPOL.PSPOLFIL import PSPOLFIL as ps_py

from DUAP.PSPOL.PSPOLFIL import generate_pav

import time
import numpy as np
import gdal

n=1500
HH = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HH_complex.tif") 
hh = HH.ReadAsArray(xoff=5000, yoff=5000, xsize=n, ysize=n)   
hh = hh[0] + 1j*hh[1]
HV = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HV_complex.tif")       
hv = HV.ReadAsArray(xoff=5000, yoff=5000, xsize=n, ysize=n)    
hv = hv[0] + 1j*hv[1]


    
sar =   np.stack( [hh, hv], axis=2 )
sar = np.moveaxis(sar, 2,0)

pow = np.absolute(sar/4.370677e+03)**2
totpow = np.sum(pow, axis=0)   
 
cov = np.atleast_3d(sar[0,:,:] * np.conj(sar[1,:,:]))
cov = np.moveaxis(cov, 2,0)

k =   np.concatenate( [np.absolute(sar),cov], axis=0 )

#out1 = np.asarray(ps_py(img=np.absolute(sar), P=totpow, NUMLK=1, WINSIZE=5))
out2 = np.asarray(ps_c(img=np.absolute(sar), P=totpow, NUMLK=1, WINSIZE=5))
compout = np.asarray(ps_c_com(img=k, P=totpow, NUMLK=1, WINSIZE=5))
    
import matplotlib.pyplot as plt
def rescale(img):
    res = np.array(img, dtype=np.float32)
    p = np.percentile(img, (2,98))
    res = img / (p[1] / 255)
    res[np.greater(res, 255)] = 255
    return(np.array(res, dtype='int16'))

rgb0 = np.concatenate((rescale(pow[[0],:,:]), rescale(pow[[1],:,:]), rescale(pow[[0],:,:])), axis=0)  
rgb0 = np.moveaxis(rgb0, 0, 2)    
rgb = np.concatenate((rescale(np.absolute(compout[[0],:,:])), rescale(np.absolute(compout[[1],:,:])), rescale(np.angle(compout[[2],:,:]))), axis=0) 
rgb = np.moveaxis(rgb, 0, 2)    
    
# fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
# hh.imshow(rescale(np.absolute(out1[0,:,:])), cmap='gray') # hsv is cyclic, like angles
# hh.set_title('HH')
# hh.set_axis_off()
# p1.imshow(rescale(np.absolute(out2[0,:,:])),  cmap='gray') # hsv is cyclic, like angles
# p1.set_title('filtered HH')
# p1.set_axis_off()
# fig.show()

fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
hh.imshow(rescale(np.absolute(pow[0,:,:])), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
p1.imshow(rescale(np.angle(compout[2,:,:])),  cmap='gray') # hsv is cyclic, like angles
p1.set_title('filtered HH')
p1.set_axis_off()
fig.show()


fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
hh.imshow(rescale(rgb0)) # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
p1.imshow(rescale(rgb)) # hsv is cyclic, like angles
p1.set_title('filtered HH')
p1.set_axis_off()
fig.show()

############
fig, (hh, p1,p2) = plt.subplots(1, 3, figsize=(15, 15))
hh.imshow(rescale(np.absolute(out2[0,:,:])), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
p1.imshow(rescale(np.absolute(cov)),  cmap='gray') # hsv is cyclic, like angles
p1.set_title('abs(complex cov)')
p1.set_axis_off()
p2.imshow(rescale(np.angle(cov)),  cmap='gray') # hsv is cyclic, like angles
p2.set_title('filtered HH')
p2.set_axis_off()
fig.show()
