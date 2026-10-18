import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype']=42
import realistic as R
fig,ax=plt.subplots(1,3,figsize=(10.5,3.3))
tt=np.linspace(0,1.07,600)
ax[0].plot(R.T_exp*12,R.fut_px,'o',ms=4,label='futures $F_{T_k}(0)$')
ax[0].plot(tt*12,R.S0*np.exp(R.curve.M(tt)),'-',lw=1,label='model forward $S_0e^{M(t)}$')
ax[0].set_xlabel('expiry (months)'); ax[0].set_ylabel('price (\\$/MMBtu)'); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
ax[0].set_title('(a) futures curve',fontsize=9)
ax[1].plot(tt*12,R.sig_atm(tt),'-',label='instantaneous ATM vol $\\sigma_{\\rm ATM}(t)$')
ax[1].plot(tt[1:]*12,np.sqrt(R.theta_atm(tt[1:])/tt[1:]),'--',label='ATM implied vol')
ax[1].set_xlabel('time (months)'); ax[1].legend(fontsize=7.5,loc='lower left'); ax[1].grid(alpha=.3); ax[1].set_title('(b) ATM volatility',fontsize=9)
C=json.load(open('res_e8calib.json')); k=np.linspace(-0.45,0.45,100)
for t,c in zip([0.25,0.5,1.0],['C0','C1','C2']):
    ax[2].plot(k,R.implied_vol(k,t),'-',color=c,lw=1,label=f'SSVI, $t={t}$')
    pts=[x for x in C if x['t']==t]; ax[2].plot([x['k'] for x in pts],[x['iv'] for x in pts],'o',color=c,mfc='none',ms=5)
ax[2].set_xlabel('log-moneyness $\\log(K/F)$'); ax[2].set_ylabel('implied volatility'); ax[2].legend(fontsize=7.5); ax[2].grid(alpha=.3)
ax[2].set_title('(c) smiles: target and PDE (circles)',fontsize=9)
plt.tight_layout(); plt.savefig('paper/fig_market.pdf')
