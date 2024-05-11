import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
import MIMO

ITER = 20000;
K = 10; # number of users
Mv = np.arange(20,520,30); # number of BS antennas
Eu_dB = 10;  Eu = 10**(Eu_dB/10);
rate_MRC = np.zeros(len(Mv)) ;
bound_MRC = np.zeros(len(Mv));
rate_ZF = np.zeros(len(Mv));

beta = MIMO.Dmatrix(K);
sqrtD = np.diag(np.sqrt(beta));





rate_MRC = rate_MRC/ITER;
bound_MRC = bound_MRC/ITER;
rate_ZF = rate_ZF/ITER;


plt.plot(Mv, rate_MRC,'g-');
plt.plot(Mv, bound_MRC,'rs');
plt.plot(Mv, rate_ZF,'b^-');
plt.grid(1,which='both')
plt.legend(["MRC", "MRC Bound", "ZF"], loc ="lower right");
plt.suptitle('SINR for MRC and ZF Receivers')
plt.ylabel('SNR ')
plt.xlabel('Number of antennas M') 

