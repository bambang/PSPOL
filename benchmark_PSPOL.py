import pkg_resources
from DUAP.PSPOL.pspol import PSPOLFIL as ps_c
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

print("Running Filter")

cProfile.runctx("np.asarray(ps_c(img=np.absolute(sar), P=ptot, NUMLK=1, WINSIZE=5))", globals(), locals(), PROF_FILE)

s = pstats.Stats(PROF_FILE)
s.strip_dirs().sort_stats("time").print_stats()
