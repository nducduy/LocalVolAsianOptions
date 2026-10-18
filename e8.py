"""Section 5.5: calendar-strip APO under an SSVI/Dupire local volatility (realistic.py)."""
import sys, json, time
import numpy as np
from apo import pde_price, mc_price, Curve
from bench import black_iv, levy_mm, curran
import realistic as R
from realistic import S0, curve, r, tmon, lam, T_pay, sigma_lv
def save(name,obj): json.dump(obj,open(f'res_{name}.json','w'),indent=1)
part=sys.argv[1]
A0=float(np.sum(lam*S0*np.exp(curve.M(tmon))))
Ks=[round(A0*m,2) for m in [0.8,0.9,1.0,1.1,1.2]]
if part=='calib':
    out=[]
    for t in [0.25,0.5,1.0]:
        Fs=S0*np.exp(curve.M(t))
        for k in [-0.3,-0.15,0.0,0.15,0.3]:
            K=Fs*np.exp(k)
            p=pde_price(np.array([t]),np.array([1.0]),K,S0,t,r,curve,sigma_lv,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=0.5)
            iv=black_iv(p,Fs,K,t,np.exp(-r*t)); tgt=float(R.implied_vol(k,t))
            out.append(dict(t=t,k=k,iv=iv,target=tgt)); print(t,k,round(iv,4),round(tgt,4),flush=True)
    save('e8calib',out)
elif part=='conv':
    out=[]
    for proxy in ['spot','bridge']:
        for Ny,Nmin in [(500,8),(1000,8),(2000,8),(1000,4),(1000,16),(1000,25),(2000,25),(4000,25)]:
            for K in [Ks[0],Ks[2],Ks[4]]:
                t0=time.time(); p=pde_price(tmon,lam,K,S0,T_pay,r,curve,sigma_lv,Ny=Ny,dt_max=1/Ny,Nmin=Nmin,sig_ref=0.5,proxy=proxy,bridge_freeze=True); c=time.time()-t0
                out.append(dict(proxy=proxy,Ny=Ny,Nmin=Nmin,K=K,price=p,cpu=c)); print(out[-1],flush=True)
    save('e8conv',out)
elif part=='price':
    out=[]
    for K in Ks:
        row=dict(K=K)
        for proxy in ['spot','bridge']:
            t0=time.time(); row[proxy]=pde_price(tmon,lam,K,S0,T_pay,r,curve,sigma_lv,Ny=2000,dt_max=1/2000,Nmin=8,sig_ref=0.5,proxy=proxy,bridge_freeze=True); row['cpu_'+proxy]=time.time()-t0
        # market-practice approximations with the SSVI implied vols of each fixing's contract
        iv=np.array([float(R.implied_vol(np.log(K/f),t)) for f,t in zip(R.fut_px[R.front],tmon)])
        row['mm_smile']=float(levy_mm(tmon,lam,K,S0,T_pay,r,curve,iv))
        row['curran_smile']=float(curran(tmon,lam,K,S0,T_pay,r,curve,float(np.sqrt(np.mean(iv**2)))))
        row['mm_atm']=float(levy_mm(tmon,lam,K,S0,T_pay,r,curve,R.sig_atm(tmon)))
        out.append(row); print(row,flush=True)
    save('e8price',out)
elif part.startswith('mc'):
    spec=part[2:].split('_'); sub=int(spec[0]); seed=81+sub if len(spec)==1 else int(spec[1])
    t0=time.time()
    m=mc_price(tmon,lam,Ks,S0,T_pay,r,curve,sigma_lv,0.48,npaths=2_000_000,chunk=250_000,sub_dt=1/(252*sub),seed=seed)
    save(f'e8{part}',dict(K=Ks,mc=list(m[:,0]),se=list(m[:,1]),cpu=time.time()-t0,sub=sub)); print(m,time.time()-t0)
