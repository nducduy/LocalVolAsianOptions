from common import *
r=0.05; cur=Curve.flat(0.05); S0=100; T=1; tm=np.arange(1,13)/12; lam=np.ones(12)/12
out=[]; Ks=[85,100,115]
for sa in [0.2,0.4]:
  for beta in [0.25,0.5,0.75]:
    sf=cev(beta,sa,S0)
    mc=mc_price(tm,lam,Ks,S0,T,r,cur,sf,sa,npaths=2_000_000,sub_dt=1/504,seed=31)
    for k,K in enumerate(Ks):
        t0=time.time(); ps=pde_price(tm,lam,K,S0,T,r,cur,sf,Ny=2000,dt_max=1/2000,sig_ref=sa); c1=time.time()-t0
        t0=time.time(); pb=pde_price(tm,lam,K,S0,T,r,cur,sf,Ny=2000,dt_max=1/2000,sig_ref=sa,proxy='bridge'); c2=time.time()-t0
        pg=pde_price(tm,lam,K,S0,T,r,cur,const(sa),Ny=2000,dt_max=1/2000)
        out.append(dict(sa=sa,beta=beta,K=K,spot=ps,bridge=pb,gbm=pg,mc=float(mc[k,0]),se=float(mc[k,1]),c1=c1,c2=c2))
        print({a:(round(b,5) if isinstance(b,float) else b) for a,b in out[-1].items()},flush=True)
save('e3',out)
