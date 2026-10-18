"""Comparison with competing methods (Section 5.4): work-precision under GBM, and
smile-adjusted moment matching under CEV."""
import sys
from common import *; from bench import *
part=sys.argv[1]
r=0.05; cur=Curve.flat(0.05); S0=100; T=1; tm=np.arange(1,13)/12; lam=np.ones(12)/12
if part=='wp':
    E2=json.load(open('res_e2.json')); Ks=[90,100,110]
    ref={K:(4*E2[str(K)][-1][1]-E2[str(K)][-2][1])/3 for K in Ks}
    out={'ref':ref,'pde':[],'rich':[],'andr':[],'mc':[],'closed':{}}
    prev=None
    for Ny in [125,250,500,1000,2000,4000]:
        t0=time.time(); v={K:pde_price(tm,lam,K,S0,T,r,cur,const(0.2),Ny=Ny,dt_max=1/Ny) for K in Ks}; c=time.time()-t0
        out['pde'].append(dict(Ny=Ny,cpu=c,err={K:v[K]-ref[K] for K in Ks}))
        if prev:
            vr={K:(4*v[K]-prev[0][K])/3 for K in Ks}
            out['rich'].append(dict(Ny=Ny,cpu=c+prev[1],err={K:vr[K]-ref[K] for K in Ks}))
        prev=(v,c); print('pde',Ny,c,flush=True)
    for Ny in [250,500,1000,2000,4000,8000]:
        t0=time.time(); v={K:andreasen_pde(tm,lam,K,S0,T,r,cur,const(0.2),Ny=Ny,dt_max=1/Ny) for K in Ks}; c=time.time()-t0
        out['andr'].append(dict(Ny=Ny,cpu=c,err={K:v[K]-ref[K] for K in Ks})); print('andr',Ny,c,flush=True)
    for N in [10_000,40_000,160_000,640_000,2_560_000]:
        t0=time.time(); m=mc_price(tm,lam,Ks,S0,T,r,cur,None,0.2,npaths=N,chunk=min(N,320_000),gbm_exact=True,seed=7); c=time.time()-t0
        out['mc'].append(dict(N=N,cpu=c,err={K:float(m[k,0]-ref[K]) for k,K in enumerate(Ks)},se={K:float(m[k,1]) for k,K in enumerate(Ks)})); print('mc',N,c,flush=True)
    for nm,f in [('vorst',vorst),('curran',curran),('levy',levy_mm)]:
        t0=time.time(); v={K:f(tm,lam,K,S0,T,r,cur,0.2) for K in Ks}; c=time.time()-t0
        out['closed'][nm]=dict(cpu=c,err={K:v[K]-ref[K] for K in Ks})
    save('e7wp',out)
elif part=='gbm':     # all cases of Table 1 and Table 4 (GBM), closed-form competitors
    E1=json.load(open('res_e1.json')); res=[]
    for d in E1:
        res.append(dict(sig=d['sig'],K=d['K'],curran=curran(tm,lam,d['K'],S0,T,r,cur,d['sig']),levy=levy_mm(tm,lam,d['K'],S0,T,r,cur,d['sig'])))
    E4=json.load(open('res_e4.json')); apo=[]
    for d in E4['gbm']:
        apo.append(dict(K=d['K'],curran=curran(tmon_b,lam_b,d['K'],S0_b,T_b,r_b,curve_b,0.35),levy=levy_mm(tmon_b,lam_b,d['K'],S0_b,T_b,r_b,curve_b,0.35)))
    save('e7gbm',dict(asian=res,apo=apo)); print(res,apo)
elif part=='cev':     # smile-adjusted moment matching (market practice) vs PDE under CEV
    res=[]
    for sa in [0.2,0.4]:
        for beta in [0.25,0.5,0.75]:
            sf=cev(beta,sa,S0)
            for K in [85,100,115]:
                iv=implied_vols(tm,K,S0,r,cur,sf,sa)
                res.append(dict(sa=sa,beta=beta,K=K,iv=list(iv),mm_smile=levy_mm(tm,lam,K,S0,T,r,cur,iv),
                                mm_atm=levy_mm(tm,lam,K,S0,T,r,cur,sa),curran_smile=None))
                print(res[-1]['sa'],beta,K,res[-1]['mm_smile'],flush=True)
    apo=[]; sf=cev(0.5,0.35,S0_b)
    for K in [72,76,80,84,88]:
        iv=implied_vols(tmon_b,K,S0_b,r_b,curve_b,sf,0.35)
        apo.append(dict(K=K,iv=list(iv),mm_smile=levy_mm(tmon_b,lam_b,K,S0_b,T_b,r_b,curve_b,iv)))
        print(K,apo[-1]['mm_smile'],flush=True)
    save('e7cev',dict(asian=res,apo=apo))
elif part=='mctime':  # CPU of the CEV Monte Carlo benchmark (per 10^5 paths, 3 strikes, step 1/1008)
    sf=cev(0.5,0.4,S0); t0=time.time()
    m=mc_price(tm,lam,[85,100,115],S0,T,r,cur,sf,0.4,npaths=200_000,sub_dt=1/1008,seed=9)
    c=time.time()-t0; save('e7mct',dict(cpu_per_1e5=c/2,se=list(m[:,1]),N=200_000)); print(c,m)
if part=='curransmile':   # Curran with a flat vol per strike = rms of the vanilla implied vols at K
    d=json.load(open('res_e7cev.json'))
    for x in d['asian']:
        s=float(np.sqrt(np.mean(np.square(x['iv'])))); x['curran_smile']=curran(tm,lam,x['K'],S0,T,r,cur,s)
    for x in d['apo']:
        s=float(np.sqrt(np.mean(np.square(x['iv'])))); x['curran_smile']=curran(tmon_b,lam_b,x['K'],S0_b,T_b,r_b,curve_b,s)
    save('e7cev',d)
if part=='apocontract':   # Brent APO: implied vol of the fixing's futures contract at strike K
    d=json.load(open('res_e7cev.json')); sf=cev(0.5,0.35,S0_b)
    for x in d['apo']:
        K=x['K']; iv=[]
        for t,Tj_ in zip(tmon_b,Tj):
            Ks=K*np.exp(-(curve_b.M(Tj_)-curve_b.M(t)))          # same forward moneyness on S_t
            p=pde_price(np.array([t]),np.array([1.0]),Ks,S0_b,t,r_b,curve_b,sf,Ny=2000,dt_max=1/2000,Nmin=25,sig_ref=0.35)
            iv.append(black_iv(p,S0_b*np.exp(curve_b.M(t)),Ks,t,np.exp(-r_b*t)))
        iv=np.array(iv); x['iv_spot']=x['iv']; x['iv']=list(iv)
        x['mm_smile_spot']=x['mm_smile']; x['curran_smile_spot']=x['curran_smile']
        x['mm_smile']=float(levy_mm(tmon_b,lam_b,K,S0_b,T_b,r_b,curve_b,iv))
        x['curran_smile']=float(curran(tmon_b,lam_b,K,S0_b,T_b,r_b,curve_b,float(np.sqrt(np.mean(iv**2)))))
        print(K,x['mm_smile']-x['mm_smile_spot'],x['curran_smile']-x['curran_smile_spot'],flush=True)
    save('e7cev',d)
