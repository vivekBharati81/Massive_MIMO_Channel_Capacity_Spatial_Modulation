
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
from scipy.special import comb
import MIMO

blockLength = 1000; 
nBlocks = 10000; 
r = 2; 
t = 2; 
EbdB = np.arange(1.0,33.1,4.0); 
Eb = 10**(EbdB/10); 
No = 1; 
Es = 2*Eb; 
SNR = Es/No; 
SNRdB = 10*np.log10(SNR); 
BER_ZF = np.zeros(len(EbdB)); 
BER_LMMSE = np.zeros(len(EbdB)); 
BERt = np.zeros(len(EbdB)); 


for blk in range(nBlocks):  
    

    for K in range(len(SNRdB)):
        
        
        BER_LMMSE[K] = BER_LMMSE[K] + np.sum(DecBitsI_LMMSE != BitsI) \
            + np.sum(DecBitsQ_LMMSE != BitsQ);


BER_ZF = BER_ZF/blockLength/nBlocks/2/t; 
BER_LMMSE = BER_LMMSE/blockLength/nBlocks/2/t; 
L=r-t+1;    BERt = comb(2*L-1, L)/2**L/SNR**L; 

plt.yscale('log')
plt.plot(SNRdB, BER_ZF,'g-');
plt.plot(SNRdB, BER_LMMSE,'b-.s');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for MIMO Channel')
plt.legend(["ZF","LMMSE", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 

