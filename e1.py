from common import *
r=0.05; cur=Curve.flat(0.05); S0=100; T=1; n=12; tm=np.arange(1,n+1)/n; lam=np.ones(n)/n
def pooled(Ks, seeds, **kw):
    R=np.array([mc_price(Ks=Ks, seed=s, **kw) for s in seeds])
    return R[:,:,0].mean(0), np.sqrt((R[:,:,1]**2).sum(0))/len(seeds)
out=[]
for sig in [0.1,0.2,0.3,0.4]:
    Ks=[85,100,115]
    m,se=pooled(Ks,[101,102,103,104],tmon=tm,lam=lam,S0=S0,T=T,r=r,curve=cur,sigma=None,sig_cv=sig,npaths=4_000_000,chunk=500_000,gbm_exact=True)
    for k,K in enumerate(Ks):
        t0=time.time(); p=pde_price(tm,lam,K,S0,T,r,cur,const(sig),Ny=2000,dt_max=1/2000); ct=time.time()-t0
        out.append(dict(sig=sig,K=K,pde=p,cpu=ct,vorst=float(vorst(tm,lam,K,S0,T,r,cur,sig)),mc=float(m[k]),se=float(se[k])))
        print(sig,K,round(p,6),round(m[k],6),round(se[k],6),round((p-m[k])/se[k],2))
save('e1',out)
