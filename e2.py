from common import *
r=0.05; cur=Curve.flat(0.05); S0=100; T=1; tm=np.arange(1,13)/12; lam=np.ones(12)/12
out={}
for K in [90,100,110]:
    rows=[]
    for Ny in [250,500,1000,2000,4000,8000]:
        t0=time.time(); p=pde_price(tm,lam,K,S0,T,r,cur,const(0.2),Ny=Ny,dt_max=1/Ny); rows.append((Ny,p,time.time()-t0))
    out[K]=rows; print(K,rows)
save('e2',out)
