import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import matplotlib.pyplot as plt
import MIMO

ITER = 8000;
K = 10; # number of users
Mv = np.arange(20,1000,60); # number of BS antennas
Eu_dB = 10;  Eu = 10**(Eu_dB/10);
rate_MRC = np.zeros(len(Mv)) ;
bound_MRC = np.zeros(len(Mv));
rate_ZF = np.zeros(len(Mv));

beta = MIMO.Dmatrix(K);
sqrtD = np.diag(np.sqrt(beta));

dftmtx = MIMO.DFTmat(K);

for it in range(ITER):
    print(it)
    for mx in range(len(Mv)):
        M = Mv[mx];
        pu = Eu; #no power scaling
        #pu = Eu/M; # power scaling
        #pu = Eu/np.sqrt(M); #power scaling
        Pp = K*pu;
        H = (nr.normal(size=(M,K))+1j*nr.normal(size=(M,K)))/np.sqrt(2);
        G = np.matmul(H, sqrtD);
        g0 = G[:,0];
        Phi = np.sqrt(1/K)*dftmtx;
        N = (nr.normal(size=(M,K))+1j*nr.normal(size=(M,K)))*np.sqrt(2);
        RxBlk = np.sqrt(1/Pp)*np.matmul(G,Phi)+N;
        Ghat = np.sqrt(1/Pp)*np.matmul(RxBlk, MIMO.H(Phi));
        g0hat = Ghat[:,0];
        g0 = G[:,0];
        e0 = g0hat - g0
        nr_MRC = pu*nl.norm(g0)**2;
        nr_bound_MRC = pu*M*beta[0];
        dr_bound_MRC = 1/K + (beta[0]+1/K/pu)/beta[0];
        g0norm = g0/nl.norm(g0);
        g0hat_norm = g0hat/nl.norm(g0);
        CSIint = np.matmul(MIMO.H(g0norm),e0);
        CSIint = pu*np.abs(CSIint)**2;
        nint = nl.norm(g0hat)**2/nl.norm(g0)**2;
        dr_MRC = CSIint + nint;
        dr_bound_MRC = dr_bound_MRC + pu*np.sum((beta[0]+1/K/pu)*beta[1:]/beta[0]);
        MUint = np.matmul(MIMO.H(g0hat_norm),G[:,1:]);
        dr_MRC = dr_MRC + pu*nl.norm(MUint)**2;
        rate_MRC[mx] = rate_MRC[mx] + np.log2(1 + nr_MRC/dr_MRC);
        bound_MRC[mx] = bound_MRC[mx] + np.log2(1+nr_bound_MRC/dr_bound_MRC);
        
    
        
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



