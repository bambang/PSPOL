from DUAP.PSPOL.pspol import mu as mu_c
from DUAP.PSPOL.pspol import nu as nu_c


import numpy as np

from DUAP.PSPOL.PSPOLFIL import Fk, Wd, mu, nu

ex = np.array([99, 105,124,138,128,34,62,
               105,91,140,98,114,63,31,
               107,94,128,138,96,61,82,
               137,129,136,105,100,55,85,
               144,145,113,132,119,39,50,
               102,97,102,110,103,34,53,
               107,146,115,123,101,76,56], dtype='float64').reshape((7,7))   
               
 
 
               
#mu(P, i, j, F, n, N2)  
F4 = np.array(Fk(4,7), dtype='int32')
Mu_c =  mu_c(P=ex, i=int(3), j=int(3), F=F4, n=int(3), N2=int(56/2))
Mu = mu(P=ex, i=int(3), j=int(3), F=F4, n=int(3), N2=int(56/2))
print(Mu, Mu_c)
Nu_c = nu_c(P=ex, i=int(3), j=int(3), F=F4, n=int(3), N2=int(56/2), mu=Mu)
Nu = nu(P=ex, i=int(3), j=int(3), F=F4, n=int(3), N2=int(56/2), mu=Mu)
print(Nu, Nu_c)