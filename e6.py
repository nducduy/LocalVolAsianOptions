from common import *
from scipy.special import gammaincc
res={}
# put route vs parity
r=0.05; cur=Curve.flat(0.05); S0=100; tm=np.arange(1,13)/12; lam=np.ones(12)/12
pc=[]
for sf,lab,px in [(const(0.3),'gbm','spot'),(cev(0.5,0.4,S0),'cev','spot'),(cev(0.5,0.4,S0),'cev-bridge','bridge')]:
  for K in [85,100,115]:
    c=pde_price(tm,lam,K,S0,1,r,cur,sf,Ny=2000,dt_max=1/2000,sig_ref=0.3,proxy=px)
    p=pde_price(tm,lam,K,S0,1,r,cur,sf,Ny=2000,dt_max=1/2000,sig_ref=0.3,proxy=px,put=True)
    pc.append((lab,K,c,p,c-p-forward_value(tm,lam,K,S0,1,r,cur)))
    print(pc[-1])
res['putcall']=pc
# absorption probabilities (time-changed driftless CEV)
ab=[]
# time change tau(T) = int_0^T exp(-2(1-beta) M(s)) ds, M(s) = int_0^s mu (Remark 2.3)
for (sa,b,S0_,cur_,T) in [(s,b,100,Curve.flat(0.05),1.0) for s in [0.2,0.4] for b in [0.25,0.5,0.75]]+[(0.35,0.5,S0_b,curve_b,T_b)]:
    sig=sa*S0_**(1-b); tg=np.linspace(0,T,4001)
    tau=np.trapezoid(np.exp(-2*(1-b)*cur_.M(tg)),tg)
    p=gammaincc(1/(2*(1-b)), S0_**(2*(1-b))/(2*sig**2*(1-b)**2*tau)); ab.append((sa,b,p)); print(sa,b,p)
res['absorb']=ab
# skew figure data
Ks=np.arange(70,93,2.0); sk={}
pg=[pde_price(tmon_b,lam_b,K,S0_b,T_b,r_b,curve_b,const(0.35),Ny=1000,dt_max=1/1000,Nmin=25) for K in Ks]
for b in [0.25,0.5,0.75]:
    pb=[pde_price(tmon_b,lam_b,K,S0_b,T_b,r_b,curve_b,cev(b,0.35,S0_b),Ny=1000,dt_max=1/1000,Nmin=25,sig_ref=0.35,proxy='bridge') for K in Ks]
    sk[b]=list(np.array(pb)-np.array(pg)); print(b,np.round(sk[b],4))
res['skew']=dict(K=list(Ks),diff={str(b):v for b,v in sk.items()},fwd=float(np.sum(lam_b*S0_b*np.exp(curve_b.M(tmon_b)))))
save('e6',res)
