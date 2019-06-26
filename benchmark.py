import pkg_resources
from DUAP.PSPOL.pspol import pspolfil as ps_c
import numpy as np
import time
import pstats, cProfile

DATA_PATH = pkg_resources.resource_filename('DUAP', 'data/')
RS2 = pkg_resources.resource_filename('DUAP', 'PSPOL/data/RS2.npy')
PROF_FILE = pkg_resources.resource_filename('DUAP', 'PSPOL/data/Profile.prof')

sar = np.load(RS2)
approximate_calibration = 4.370677e+03
power = (np.absolute(sar) / approximate_calibration)**2
ptot = np.sum(power, axis=0) 

print("Running Filter on 500 x 500 array with window size 7")

cProfile.runctx("ps_c(img=np.absolute(sar), P=ptot, numlook=1, winsize=7)", globals(), locals(), PROF_FILE)

s = pstats.Stats(PROF_FILE)
s.strip_dirs().sort_stats("time").print_stats()
