"""Section 5.7: CPU times (Python/NumPy/SciPy, one core). Run alone on an idle machine."""
import numpy as np, time, json
from common import *
from greeks import grid_greeks
import realistic as R


def best(f, rep=3):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); v = f(); t.append(time.perf_counter() - t0)
    return v, min(t)


def nsteps(tmon, Ny, Nmin, rann=2):
    tg = np.concatenate([[0.0], tmon]); d = np.diff(tg); d = d[d > 1e-14]
    return int(sum(max(Nmin, int(np.ceil(x * Ny))) + rann - 1 for x in d))


r = 0.05; cur = Curve.flat(0.05); S0 = 100; tm = np.arange(1, 13) / 12; lm = np.ones(12) / 12
sfA = cev(0.5, 0.4, S0); sfB = cev(0.5, 0.35, S0_b)
cases = [
    ('Asian, 12 monthly', 'GBM', lambda Ny, **k: pde_price(tm, lm, 100, S0, 1, r, cur, const(0.4), Ny=Ny, dt_max=1 / Ny, **k), tm, 4),
    ('Asian, 12 monthly', 'CEV spot', lambda Ny, **k: pde_price(tm, lm, 100, S0, 1, r, cur, sfA, Ny=Ny, dt_max=1 / Ny, sig_ref=0.4, **k), tm, 4),
    ('Asian, 12 monthly', 'CEV bridge', lambda Ny, **k: pde_price(tm, lm, 100, S0, 1, r, cur, sfA, Ny=Ny, dt_max=1 / Ny, sig_ref=0.4, proxy='bridge', **k), tm, 4),
    ('Brent APO, 21 daily', 'GBM', lambda Ny, **k: pde_price(tmon_b, lam_b, 80, S0_b, T_b, r_b, curve_b, const(0.35), Ny=Ny, dt_max=1 / Ny, Nmin=25, **k), tmon_b, 25),
    ('Brent APO, 21 daily', 'CEV spot', lambda Ny, **k: pde_price(tmon_b, lam_b, 80, S0_b, T_b, r_b, curve_b, sfB, Ny=Ny, dt_max=1 / Ny, Nmin=25, sig_ref=0.35, **k), tmon_b, 25),
    ('Brent APO, 21 daily', 'CEV bridge', lambda Ny, **k: pde_price(tmon_b, lam_b, 80, S0_b, T_b, r_b, curve_b, sfB, Ny=Ny, dt_max=1 / Ny, Nmin=25, sig_ref=0.35, proxy='bridge', bridge_freeze=True, **k), tmon_b, 25),
    ('Calendar strip, 252 daily', 'LV spot', lambda Ny, **k: pde_price(R.tmon, R.lam, 3.38, R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, Ny=Ny, dt_max=1 / Ny, Nmin=8, sig_ref=0.5, **k), R.tmon, 8),
    ('Calendar strip, 252 daily', 'LV bridge', lambda Ny, **k: pde_price(R.tmon, R.lam, 3.38, R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, Ny=Ny, dt_max=1 / Ny, Nmin=8, sig_ref=0.5, proxy='bridge', bridge_freeze=True, **k), R.tmon, 8),
]
out = []
for prod, model, f, tmon_, Nmin in cases:
    row = dict(product=prod, model=model)
    for Ny in [1000, 2000]:
        _, c = best(lambda: f(Ny)); row[f'cpu{Ny}'] = c; row[f'steps{Ny}'] = nsteps(tmon_, Ny, Nmin)
    out.append(row); print(row, flush=True)

# GBM: one solve gives the whole strike strip (the PDE for u does not involve K)
Ks = [80, 90, 100, 110, 120]; Ny = 2000
y0 = -max(Ks) / S0 * np.exp(6 * 0.4)
(p, y, u), c1 = best(lambda: pde_price(tm, lm, 100, S0, 1, r, cur, const(0.4), Ny=Ny, dt_max=1 / Ny, y0=y0, return_grid=True))
from scipy.interpolate import CubicSpline
cs = CubicSpline(y, u); strip = [S0 * float(cs(-K / S0)) for K in Ks]
indiv = [pde_price(tm, lm, K, S0, 1, r, cur, const(0.4), Ny=Ny, dt_max=1 / Ny) for K in Ks]
gstrip = dict(cpu_one=c1, maxdiff=max(abs(a - b) for a, b in zip(strip, indiv)), Ks=Ks)
print(gstrip)
# grid Greeks cost relative to a price
_, cg = best(lambda: grid_greeks(tmon_b, lam_b, 80, S0_b, T_b, r_b, curve_b, sfB, Ny=1000, dt_max=1 / 1000, Nmin=25, sig_ref=0.35))
# Monte Carlo cost (log-Euler, 4 steps per business day, antithetic + control variate), 2e5 paths
mc = {}
for nm, f in [('Asian CEV', lambda: mc_price(tm, lm, [85, 100, 115], S0, 1, r, cur, sfA, 0.4, npaths=200_000, sub_dt=1 / 1008, seed=5)),
              ('Brent APO CEV', lambda: mc_price(tmon_b, lam_b, [72, 76, 80, 84, 88], S0_b, T_b, r_b, curve_b, sfB, 0.35, npaths=200_000, sub_dt=1 / 1008, seed=5)),
              ('Calendar strip LV', lambda: mc_price(R.tmon, R.lam, [2.7, 3.04, 3.38, 3.72, 4.06], R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, 0.48, npaths=200_000, sub_dt=1 / 1008, seed=5))]:
    m, c = best(f, rep=1); mc[nm] = dict(cpu_2e5=c, se=list(m[:, 1])); print(nm, mc[nm], flush=True)
json.dump(dict(cases=out, gbm_strip=gstrip, mc=mc, cpu_grid_greeks_brent_cev_spot_1000=cg), open('res_e10time.json', 'w'), indent=1)
