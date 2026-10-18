import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype']=42
d=json.load(open('res_e7wp.json')); P='paper/'
mx=lambda e: max(abs(v) for v in e.values())
plt.figure(figsize=(5.4,3.8))
for key,lab,mk in [('pde','PDE (this paper)','o-'),('rich','PDE + Richardson','s-'),('andr','PDE with jump conditions','^--')]:
    plt.loglog([x['cpu'] for x in d[key]],[mx(x['err']) for x in d[key]],mk,ms=5,label=lab)
A=d['andr']; ra=[(A[i]['cpu']+A[i-1]['cpu'], max(abs((4*A[i]['err'][K]-A[i-1]['err'][K])/3) for K in A[i]['err'])) for i in range(1,len(A))]
plt.loglog([a for a,b in ra],[b for a,b in ra],'v--',ms=5,color='C2',mfc='none',label='jump conditions + Richardson')
plt.loglog([x['cpu'] for x in d['mc']],[mx(x['se']) for x in d['mc']],'d:',ms=5,label='Monte Carlo (SE)')
for nm,mk,lab in [('curran','P','Curran'),('vorst','X','Vorst'),('levy','*','Levy')]:
    x=d['closed'][nm]; plt.loglog([max(x['cpu'],1e-3)],[mx(x['err'])],mk,ms=8,color='k' if nm=='curran' else 'gray',label=lab,linestyle='none')
plt.xlabel('CPU time for three strikes (s)'); plt.ylabel('maximum absolute error')
plt.grid(alpha=.3,which='both'); plt.legend(fontsize=7.5,ncol=2,loc='lower left'); plt.tight_layout(); plt.savefig(P+'fig_wp.pdf')
