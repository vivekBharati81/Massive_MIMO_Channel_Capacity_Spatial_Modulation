import numpy as np
import numpy.linalg as nl
import numpy.random as nr
from scipy.stats import norm
from scipy.stats import unitary_group


def Dmatrix(K):
    var_nr = (10**(8/10))**2; mean_nr = 3;
    mu_nr = np.log10(mean_nr**2/np.sqrt(var_nr+mean_nr**2)); 
    sigma_nr = np.sqrt(np.log10(var_nr/(mean_nr**2+1)));
    nr = np.random.lognormal(mu_nr,sigma_nr,K);
    dr = np.random.randint(100,1000,K)/100;
    beta = nr/dr**3.0;
    return beta;

def DFTmat(K):
    kx, lx = np.meshgrid(np.arange(K), np.arange(K))
    omega = np.exp(-2*np.pi*1j/K)
    dftmtx = np.power(omega,kx*lx)
    return dftmtx

def Q(x):
    return 1-norm.cdf(x);

def QPSK(m,n):
    return ((2*nr.randint(2,size=(m,n))-1)+1j*(2*nr.randint(2,size=(m,n))-1))/np.sqrt(2);

def H(G):
    return np.conj(np.transpose(G));

def ArrayDictionary(G,t):
    lxx = 2/G*np.arange(G)-1;
    lx, kx = np.meshgrid(lxx, np.arange(t))
    omega = np.exp(-1j*np.pi)
    dmtx = 1/np.sqrt(t)*np.power(omega,kx*lx)
    return dmtx

def RF_BB_matrices(numAnt,numRF,N_Beam):
    NBlk = numAnt/numRF;
    RFmat = 1/np.sqrt(numAnt)*DFTmat(numAnt);
    U = unitary_group.rvs(numRF);
    V = unitary_group.rvs(int(N_Beam/NBlk));
    CenterMat = np.concatenate((np.identity(int(N_Beam/NBlk)), 
                                np.zeros((int(numRF-N_Beam/NBlk),int(N_Beam/NBlk)))),axis=0);
    BB_diag = nl.multi_dot([U,CenterMat,H(V)]);
    BBmat = np.kron(np.identity(int(NBlk)),BB_diag); 
    return RFmat, BBmat

def OMP(y,Q,thrld):
    [rq,cq] = Q.shape; 
    set_I = np.zeros(cq);  
    r_prev = np.zeros((rq,1)); 
    hb_omp = np.zeros((cq,1)) + np.zeros((cq,1))*1j;
    r_curr = y; 
    Qa = np.zeros((rq,cq))+ np.zeros((rq,cq))*1j; 
    ix1 = 0;
    while np.absolute(nl.norm(r_prev)**2 - nl.norm(r_curr)**2) > thrld:
        m_ind = np.argmax(np.absolute(np.matmul(H(Q),r_curr))); 
        set_I[ix1] = m_ind;
        Qa[:,ix1] = Q[:,m_ind];
        hb_ls = np.matmul(nl.pinv(Qa[:,0:ix1+1]),y); 
        r_prev = r_curr;
        r_curr = y - np.matmul(Qa[:,0:ix1+1],hb_ls); 
        ix1 = ix1 + 1;

    set_I_nz = set_I[0:ix1];
    hb_omp[set_I_nz.astype(int)] = hb_ls;
    return hb_omp

def SOMP(Opt, Dict, Ryy, numRF):
    rq, cq = np.shape(Dict); 
    Res = Opt; 
    RF = np.zeros((rq,numRF))+1j*np.zeros((rq,numRF));   
    for iter1 in range(numRF):
        phi = nl.multi_dot([H(Dict),Ryy,Res]); 
        phi_phiH = AAH(phi); 
        m_ind = np.argmax(np.abs(np.diag(phi_phiH)));
        RF[:,iter1] = Dict[:,m_ind];  
        RFc = RF[:,0:iter1+1];
        BB = nl.multi_dot([nl.inv(nl.multi_dot([H(RFc),Ryy,RFc])),H(RFc),Ryy,Opt]);
        Res = (Opt-np.matmul(RFc,BB))/nl.norm(Opt-np.matmul(RFc,BB));    
    return  BB, RF


def SOMP_Est(y,Qbar,thrld):
    rq,cq = np.shape(Qbar);
    ry,cy = np.shape(y);
    set_I = np.zeros((cq,1));
    r_prev = np.zeros((ry,cy))+1j*np.zeros((ry,cy));
    hb_OMP = np.zeros((cq,cy))+1j*np.zeros((cq,cy));
    r_curr = y; 
    Q_a = np.zeros((rq,cq))+1j*np.zeros((rq,cq)); 
    ix1 = 0;
    while(abs((nl.norm(r_prev,2))**2 - (nl.norm(r_curr,2))**2) > thrld):
        psi = nl.multi_dot([H(Qbar),r_curr]);
        m_ind = np.argmax(np.abs(np.diag(AAH(psi))));
        set_I[ix1] = m_ind;
        Q_a[:,ix1] = Qbar[:,m_ind];
        Q_c = Q_a[:,0:ix1+1];
        Hb_LS = np.matmul(nl.pinv(Q_c),y);
        r_prev = r_curr;
        r_curr = y - np.matmul(Q_c,Hb_LS);
        ix1 = ix1 + 1;
    set_I_nz = set_I[0:ix1];    
    hb_OMP[set_I_nz.astype(int).flatten(),:] = Hb_LS;
    return hb_OMP

def MSE_time_domain(H,Ht,Fsub,r,t,Nt):
    Ht_est = np.zeros((r,t,Nt))+1j*np.zeros((r,t,Nt));
    for tx in range(t):
        for rx in range(r):
            Ht_est[rx,tx,:] = np.matmul(nl.pinv(Fsub),H[rx,tx,:]);
    MSE_td = nl.norm(Ht.flatten()-Ht_est.flatten())**2/t/r/Nt;
    return MSE_td

def SBL(y,Q,sigma_2):
    N, M = np.shape(Q);
    Gamma = np.identity(M);
    for iter in range(50):
        Sigma = nl.inv(1/sigma_2*np.matmul(H(Q),Q) + nl.inv(Gamma));
        mu = 1/sigma_2*nl.multi_dot([Sigma,H(Q),y]);
        Gamma = np.diag(np.diag(Sigma)+np.abs(mu).flatten()**2);
    return mu, Gamma 
        


def mmWaveMIMOChannelGenerator(A_R,A_T,G,L):
    t = A_T.shape[0];
    r = A_R.shape[0];
    Psi = np.zeros(shape=(t*r,L))+np.zeros(shape=(t*r,L))*1j;
    tax = nr.choice(G, L, replace=False);
    rax = nr.choice(G, L, replace=False);
    alpha = 1/np.sqrt(2)*(nr.normal(0,1,L)+1j*nr.normal(0,1,L));
    A_T_genie = A_T[:, tax];
    A_R_genie = A_R[:, rax];
    for jx in range(L):
        Psi[:,jx] = np.kron(np.conj(A_T[:,tax[jx]]),A_R[:,rax[jx]]);       
    return alpha, Psi, A_R_genie, A_T_genie


def mmWaveMIMO_OFDMChannelGenerator(A_R,A_T,L,numTaps):
    t,G = np.shape(A_T);
    r,G = np.shape(A_R);
    Ht = np.zeros((r,t,numTaps)) + 1j*np.zeros((r,t,numTaps));
    Psi = np.zeros(shape=(t*r,L))+np.zeros(shape=(t*r,L))*1j;
    tax = nr.choice(G, L, replace=False);
    rax = nr.choice(G, L, replace=False);
    A_T_genie = A_T[:, tax];
    A_R_genie = A_R[:, rax];
    for jx in range(L):
        Psi[:,jx] = np.kron(np.conj(A_T[:,tax[jx]]),A_R[:,rax[jx]]);
    for px in range(numTaps):
        alpha = 1/np.sqrt(2)*(nr.normal(0,1,L)+1j*nr.normal(0,1,L));
        Ht[:,:,px] = np.sqrt(t*r/L)*nl.multi_dot([A_R_genie,np.diag(alpha),H(A_T_genie)])
    return Ht, Psi, A_R_genie, A_T_genie



def AHA(A):
    return np.matmul(H(A),A)

def AAH(A):
    return np.matmul(A,H(A))

def mimo_capacity(Hmat, TXcov, Ncov):
    r, c = np.shape(Hmat);
    inLD = np.identity(r) + nl.multi_dot([nl.inv(Ncov),Hmat,TXcov,H(Hmat)]);
    C = np.log2(nl.det(inLD));
    return np.abs(C)


def OPT_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = 0;
    while not CAP:
        onebylam = (SNR + sum(1/S[0:t]**2))/t;
        if  onebylam - 1/S[t-1]**2 >= 0:
            optP = onebylam - 1/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP));
        elif onebylam - 1/S[t-1]**2 < 0:
            t = t-1;            
    return CAP

def EQ_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = sum(np.log2(1+ S[0:t]**2 * SNR/t));
    return CAP


def MPAM_DECODER(EqSym,M):
    DecSym = np.round((EqSym+M-1)/2);
    DecSym[np.where(DecSym<0)] = 0;
    DecSym[np.where(DecSym>(M-1))] = M-1      
    return DecSym

def MQAM_DECODER(EqSym,M):
    sqM = np.int(np.sqrt(M));
    DecSym = np.round((EqSym+sqM-1)/2);
    DecSym[np.where(DecSym<0)]=0;
    DecSym[np.where(DecSym>(sqM-1))]=sqM-1      
    return DecSym

def PHYDAS(L_f,N):
    H1=0.971960;
    H2=np.sqrt(2)/2;
    H3=0.235147;
    fh=1+2*(H1+H2+H3);
    hef=np.zeros((1,L_f+1));
    for i in range(L_f+1):
        hef[0,i]=1-2*H1*np.cos(np.pi*i/(2*N))+2*H2*np.cos(np.pi*i/N)-2*H3*np.cos(np.pi*i*3/(2*N));

    hef = hef/fh;
    p_k = hef/nl.norm(hef);
    return(p_k)

def UPSAMPLE(H,k):
    m = H.shape[0];
    n = H.shape[1];
    G = np.zeros((int(m*k),n))+1j*np.zeros((int(m*k),n));
    for ix in range(m):
        G[ix*k,:] = H[ix,:];
        
    return(G)


def DOWNSAMPLE(H,k):
    m = H.shape[0];
    n = H.shape[1];
    G = np.zeros((int(m/k),n))+1j*np.zeros((int(m/k),n));
    for ix in range(int(m/k)):
        G[ix,:] = H[ix*k,:];
        
    return(G)