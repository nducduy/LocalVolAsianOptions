"""Figure 2: error of the PDE price at N_y=1000 as a function of the strike (GBM Asian call,
sigma=0.2, settings of Table 2), measured against the Richardson extrapolation of N_y=1000, 2000."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from common import *
r = 0.05; cur = Curve.flat(0.05); S0 = 100; tm = np.arange(1, 13) / 12; lam = np.ones(12) / 12
V = lambda K, Ny: pde_price(tm, lam, K, S0, 1, r, cur, const(0.2), Ny=Ny, dt_max=1 / Ny)
K = np.arange(70, 140.1, 1.0)
e = np.array([4 / 3 * (V(k, 1000) - V(k, 2000)) for k in K])
pts = {int(k): float(e[list(K).index(k)]) for k in (90, 100, 110)}
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.axhline(0, color='0.6', lw=0.8)
ax.plot(K, e * 1e4, color='#1f4e9c', lw=1.4)
for k, c in zip((90, 100, 110), ('#2c7a4b', '#8e44ad', '#c0392b')):
    ax.plot(k, pts[k] * 1e4, 'o', color=c, ms=6, label=f'$K={k}$')
ax.set_xlabel('strike $K$'); ax.set_ylabel(r'error at $N_y=1000$ ($\times10^{-4}$)')
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout(); fig.savefig('paper/fig_conv_strike.pdf')
json.dump({'K': K.tolist(), 'err1000': e.tolist(), 'points': pts}, open('res_figstrike.json', 'w'), indent=1)
print(pts)
