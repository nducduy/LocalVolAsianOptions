"""Section 5.6: Delta and Gamma from the PDE grid, against bump-and-revalue PDE and Monte Carlo.

With V(S) = S u(0, y), y = -K/S:  Delta = u - y u_y,  Gamma = y^2 u_yy / S.
For GBM and for the spot proxy the PDE coefficients do not depend on S0, so these are the exact
derivatives of the PDE price. For the bridge proxy the coefficients depend on S0 through the
anchors c_j, and the grid Greeks hold the proxy fixed ("sticky-proxy" Greeks).
"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from common import *


def grid_greeks(tmon, lam, K, S0, T, r, curve, sigma, **kw):
    p, y, u = pde_price(tmon, lam, K, S0, T, r, curve, sigma, return_grid=True, **kw)
    cs = CubicSpline(y, u); ys = -K / S0
    return p, float(cs(ys) - ys * cs(ys, 1)), float(ys ** 2 * cs(ys, 2) / S0)


def bump_greeks(tmon, lam, K, S0, T, r, curve, sigma, h, **kw):
    f = lambda s: pde_price(tmon, lam, K, s, T, r, curve, sigma, **kw)
    vm, v0, vp = f(S0 - h), f(S0), f(S0 + h)
    return v0, (vp - vm) / (2 * h), (vp - 2 * v0 + vm) / h ** 2


def mc_greeks(tmon, lam, K, S0, T, r, curve, sigma, sig_cv, h, nbatch=8, npaths=500_000, **kw):
    """common random numbers: the same seed for S0-h, S0, S0+h; batches give the standard errors."""
    D, G, V = [], [], []
    for b in range(nbatch):
        v = [mc_price(tmon, lam, [K], s, T, r, curve, sigma, sig_cv, npaths=npaths, seed=1000 + b, **kw)[0, 0]
             for s in (S0 - h, S0, S0 + h)]
        V.append(v[1]); D.append((v[2] - v[0]) / (2 * h)); G.append((v[2] - 2 * v[1] + v[0]) / h ** 2)
    st = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a))))
    return st(V), st(D), st(G)


if __name__ == '__main__':
    part = sys.argv[1]; out = []
    if part == 'brent':
        K = 80.0; sa = 0.35
        cases = [('GBM', const(sa), 'spot', dict(gbm_exact=True)),
                 ('CEV', cev(0.5, sa, S0_b), 'spot', dict(sub_dt=1 / 1008)),
                 ('CEV', cev(0.5, sa, S0_b), 'bridge', dict(sub_dt=1 / 1008))]
        mcc = {}
        for model, sf, proxy, mkw in cases:
            kw = dict(Ny=2000, dt_max=1 / 2000, Nmin=25, sig_ref=sa, proxy=proxy, bridge_freeze=True)
            t0 = time.time(); g = grid_greeks(tmon_b, lam_b, K, S0_b, T_b, r_b, curve_b, sf, **kw); cg = time.time() - t0
            t0 = time.time(); b = bump_greeks(tmon_b, lam_b, K, S0_b, T_b, r_b, curve_b, sf, 0.005 * S0_b, **kw); cb = time.time() - t0
            if model not in mcc:
                mcc[model] = mc_greeks(tmon_b, lam_b, K, S0_b, T_b, r_b, curve_b, None if model == 'GBM' else sf, sa,
                                       0.005 * S0_b, **mkw)
            out.append(dict(model=model, proxy=proxy, grid=g, bump=b, mc=mcc[model], cpu_grid=cg, cpu_bump=cb))
            print(out[-1], flush=True)
        save('e9brent', out)
    elif part == 'strip':
        import realistic as R
        K = round(float(np.sum(R.lam * R.S0 * np.exp(R.curve.M(R.tmon)))), 2)
        for proxy in ['spot', 'bridge']:
            kw = dict(Ny=1000, dt_max=1 / 1000, Nmin=8, sig_ref=0.5, proxy=proxy, bridge_freeze=True)
            t0 = time.time(); g = grid_greeks(R.tmon, R.lam, K, R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, **kw); cg = time.time() - t0
            t0 = time.time(); b = bump_greeks(R.tmon, R.lam, K, R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, 0.005 * R.S0, **kw); cb = time.time() - t0
            out.append(dict(proxy=proxy, K=K, grid=g, bump=b, cpu_grid=cg, cpu_bump=cb)); print(out[-1], flush=True)
        m = mc_greeks(R.tmon, R.lam, K, R.S0, R.T_pay, R.r, R.curve, R.sigma_lv, 0.48, 0.005 * R.S0, nbatch=8,
                      npaths=250_000, sub_dt=1 / 1008)
        out.append(dict(mc=m)); print(m, flush=True)
        save('e9strip', out)
