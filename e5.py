import sys
from common import *
part=sys.argv[1]
if part=='prog':
    r=r_b; sa=0.35; beta=0.5; t0=116/252; rem=tmon_b>t0+1e-12; tr=tmon_b[rem]; lr=lam_b[rem]; cl=ShiftCurve(curve_b,t0)
    sf=cev(beta,sa,S0_b); sft=lambda t,S: sf(t+t0,S); out=[]
    for Fb1,F2 in [(78.0,77.6),(80.5,80.1),(83.0,82.6)]:
        Ap=theta*11*Fb1; St=F2*np.exp(-(curve_b.M(T2)-curve_b.M(t0))); Kp=80-Ap
        m=mc_price(tr,lr,[80],St,T_b,r,curve_b,sf,sa,npaths=2_000_000,sub_dt=1/8064,seed=45,t_start=t0,A_past=Ap)
        ps=pde_price(tr-t0,lr,Kp,St,T_b-t0,r,cl,sft,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=sa,proxy='spot')
        pb=pde_price(tr-t0,lr,Kp,St,T_b-t0,r,cl,sft,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=sa,proxy='bridge')
        out.append(dict(Fb1=Fb1,spot=ps,bridge=pb,mc=float(m[0,0]),se=float(m[0,1]))); print(out[-1],flush=True)
    save('e5prog',out)
elif part=='apo':
    sf=cev(0.5,0.35,S0_b); Ks=[72,76,80,84,88]
    m=mc_price(tmon_b,lam_b,Ks,S0_b,T_b,r_b,curve_b,sf,0.35,npaths=2_000_000,sub_dt=1/1008,seed=46)
    save('e5apo',[dict(K=K,mc=float(m[k,0]),se=float(m[k,1])) for k,K in enumerate(Ks)]); print(m)
else:
    sa=float(part); r=0.05; cur=Curve.flat(0.05); S0=100; tm=np.arange(1,13)/12; lam=np.ones(12)/12; out=[]
    for beta in [0.25,0.5,0.75]:
        m=mc_price(tm,lam,[85,100,115],S0,1.0,r,cur,cev(beta,sa,S0),sa,npaths=2_000_000,sub_dt=1/1008,seed=47)
        out+= [dict(sa=sa,beta=beta,K=K,mc=float(m[k,0]),se=float(m[k,1])) for k,K in enumerate([85,100,115])]
        print(out[-3:],flush=True)
    save('e5_'+part,out)
