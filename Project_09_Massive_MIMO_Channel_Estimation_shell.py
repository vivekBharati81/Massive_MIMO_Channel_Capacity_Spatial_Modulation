import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import matplotlib.pyplot as plt
import MIMO

ITER = 2000;
K = 10; # number of users
Mv = np.arange(20,1000,60); # number of BS antennas
Eu_dB = 10;  Eu = 10**(Eu_dB/10);
rate_MRC = np.zeros(len(Mv)) ;
bound_MRC = np.zeros(len(Mv));
rate_ZF = np.zeros(len(Mv));

beta = MIMO.Dmatrix(K);
sqrtD = np.diag(np.sqrt(beta));

dftmtx = MIMO.DFTmat(K);


        
rate_MRC = rate_MRC/ITER;
bound_MRC = bound_MRC/ITER; 


plt.plot(Mv, rate_MRC,'g-');
plt.plot(Mv, bound_MRC,'rs');
plt.grid(1,which='both')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0.1*y1,2*y2))
plt.legend(["MRC", "MRC Bound"], loc ="upper left");
plt.suptitle('SINR for MRC with CSI Estimation')
plt.ylabel('Rate')
plt.xlabel('Number of antennas M') 



