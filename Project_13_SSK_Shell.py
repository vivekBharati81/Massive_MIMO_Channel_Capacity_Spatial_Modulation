import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import numpy.matlib as nm

SNRdB = np.arange(0,11,2);
ITER = 100000;
Nt = 4;
Nr = 4;
bpcu = np.log2(Nt);
BERopt = np.zeros(len(SNRdB));

for ite in range(ITER):
    antIndex = nr.randint(2**bpcu);
    H = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,Nt)) + 1j*nr.normal(0,1,(Nr,Nt)));
    RxNoise = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,1)) + 1j*nr.normal(0,1,(Nr,1)));
    for K in range(len(SNRdB)):
        rho = 10**(SNRdB[K]/10);
        RxVec = np.sqrt(rho)*H[:,antIndex:antIndex+1] + RxNoise;
        
        #Optimal ML detector
        MLobj=np.sum(np.absolute(np.sqrt(rho)*H-nm.repmat(RxVec,1,Nt))**2,axis=0);
        decIndex = np.argmin(MLobj);
        BERopt[K] = BERopt[K] + (decIndex != antIndex);
    antIndex = nr.randint(2**bpcu);
        

BERopt = BERopt/(bpcu*ITER);

plt.yscale('log')
plt.plot(SNRdB, BERopt,'go-');
plt.grid(1,which='both')
plt.suptitle('BER for SSK')
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 