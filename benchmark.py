from DUAP.PSPOL.pspol import PSPOLFIL as ps_c
#from DUAP.PSPOL.PSPOLFIL import PSPOLFIL as ps_py
##from DUAP.PSPOL.PSPOLFIL import generate_pav

# import time
import numpy as np
import gdal

import pstats, cProfile

n=750
HH = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HH_complex.tif") 
hh = HH.ReadAsArray(xoff=5000, yoff=5000, xsize=n, ysize=n)   
hh = hh[0] + 1j*hh[1]
HV = gdal.Open(r"C:\Users\brownn\RS2_OK64300_PK591962_DK524798_F0W3_20150617_130940_HH_HV_SLC\imagery_HV_complex.tif")       
hv = HV.ReadAsArray(xoff=5000, yoff=5000, xsize=n, ysize=n)    
hv = hv[0] + 1j*hv[1]
    
sar =   np.stack( [hh, hv], axis=2 )
sar = np.moveaxis(sar, 2,0)
#sar = sar[:,175:225,175:225]
pow = np.absolute(sar/4.370677e+03)**2
totpow = np.sum(pow, axis=0)   
#pav = generate_pav(totpow, n=3)   

# out1 = np.asarray(ps_py(img=np.absolute(sar), P=totpow, NUMLK=1, WINSIZE=5))
cProfile.runctx("np.asarray(ps_c(img=np.absolute(sar), P=totpow, NUMLK=1, WINSIZE=5))", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

#     
# import matplotlib.pyplot as plt
# def rescale(img):
#     res = np.array(img, dtype=np.float32)
#     p = np.percentile(img, (2,98))
#     res = img / (p[1] / 255)
#     res[np.greater(res, 255)] = 255
#     return(np.array(res, dtype='int16'))
#     
# rgb = np.concatenate((rescale(out2[[0],:,:]), rescale(out2[[1],:,:]), rescale(out2[[0],:,:])), axis=0) 
# rgb = np.moveaxis(rgb, 0, 2)    
#     
# fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
# hh.imshow(rescale(np.absolute(out1[0,:,:])), cmap='gray') # hsv is cyclic, like angles
# hh.set_title('HH')
# hh.set_axis_off()
# p1.imshow(rescale(np.absolute(out2[0,:,:])),  cmap='gray') # hsv is cyclic, like angles
# p1.set_title('filtered HH')
# p1.set_axis_off()
# fig.show()
# 
# fig, (hh, p1) = plt.subplots(1, 2, figsize=(15, 15))
# hh.imshow(rescale(np.absolute(pow[0,:,:])), cmap='gray') # hsv is cyclic, like angles
# hh.set_title('HH')
# hh.set_axis_off()
# p1.imshow(rescale(np.absolute(out2[0,:,:])),  cmap='gray') # hsv is cyclic, like angles
# p1.set_title('filtered HH')
# p1.set_axis_off()
# fig.show()
# 
# 
