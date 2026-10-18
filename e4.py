from common import *
r=r_b; Ks=[72,76,80,84,88]; sa=0.35
out={'gbm':[],'cev':[],'prog':[]}
# GBM
mc=mc_price(tmon_b,lam_b,Ks,S0_b,T_b,r,curve_b,None,sa,npaths=4_000_000,chunk=500_000,gbm_exact=True,seed=41)
for k,K in enumerate(Ks):
    t0=time.time(); p=pde_price(tmon_b,lam_b,K,S0_b,T_b,r,curve_b,const(sa),Ny=2000,dt_max=1/2000,Nmin=25); c=time.time()-t0
    out['gbm'].append(dict(K=K,pde=p,cpu=c,vorst=float(vorst(tmon_b,lam_b,K,S0_b,T_b,r,curve_b,sa)),mc=float(mc[k,0]),se=float(mc[k,1])))
    print(out['gbm'][-1],flush=True)
# CEV
beta=0.5; sf=cev(beta,sa,S0_b)
mc=mc_price(tmon_b,lam_b,Ks,S0_b,T_b,r,curve_b,sf,sa,npaths=2_000_000,sub_dt=1/504,seed=42)
for k,K in enumerate(Ks):
    ps=pde_price(tmon_b,lam_b,K,S0_b,T_b,r,curve_b,sf,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=sa)
    pb=pde_price(tmon_b,lam_b,K,S0_b,T_b,r,curve_b,sf,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=sa,proxy='bridge')
    out['cev'].append(dict(K=K,spot=ps,bridge=pb,gbm=out['gbm'][k]['pde'],mc=float(mc[k,0]),se=float(mc[k,1])))
    print(out['cev'][-1],flush=True)
# in-progress at day 116 (11 fixings of F_T1 done)
t0=116/252; rem=tmon_b>t0+1e-12; tr=tmon_b[rem]; lr=lam_b[rem]
cl=ShiftCurve(curve_b,t0)
for (Fb1,F2) in [(78.0,77.6),(80.5,80.1),(83.0,82.6)]:
    Ap=theta*11*Fb1; St=F2*np.exp(-(curve_b.M(T2)-curve_b.M(t0)))
    for model in ['gbm','cev']:
        sf_ = const(sa) if model=='gbm' else cev(beta,sa,S0_b)
        sft = (lambda t,S,f=sf_: f(t+t0,S))
        mc=mc_price(tr,lr,[40,80],St,T_b,r,curve_b,sf_,sa,npaths=2_000_000,sub_dt=1/504,seed=43,
                    gbm_exact=(model=='gbm'),t_start=t0,A_past=Ap)
        for k,K in enumerate([40,80]):
            Kp=K-Ap
            if Kp<=0:
                p=float(np.exp(-r*(T_b-t0))*(Ap-K+np.sum(lr*St*np.exp(curve_b.M(tr)-curve_b.M(t0))))); how='parity'
            else:
                p=pde_price(tr-t0,lr,Kp,St,T_b-t0,r,cl,sft,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=sa,
                            proxy='bridge' if model=='cev' else 'spot'); how='pde'
            out['prog'].append(dict(model=model,Fb1=Fb1,F2=F2,K=K,Kp=Kp,price=p,how=how,mc=float(mc[k,0]),se=float(mc[k,1])))
            print(out['prog'][-1],flush=True)
save('e4',out)
