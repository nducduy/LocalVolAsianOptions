"""Section 5.5: a calendar-strip APO on a seasonal (natural-gas style) futures curve with a local
volatility obtained from an arbitrage-free SSVI implied-volatility surface by Dupire's formula.

All inputs are stylized (not market data) but chosen to have the features of a natural-gas market:
a seasonal curve with a winter premium, a seasonal ATM volatility in calendar time, and a call skew.
Business-day clock: t = d/252, pricing date d = 0 (start of October).
"""
import numpy as np
from apo import Curve

BD = 252
# 13 monthly contracts (delivery Nov, Dec, ..., Nov of the next year); contract k expires on business
# day 21k - 3, i.e. three business days before the end of the month preceding delivery.
exp_days = 21 * np.arange(1, 14) - 3
fut_px = np.array([3.10, 3.65, 4.05, 3.85, 3.40, 3.05, 3.00, 3.10, 3.30, 3.35, 3.30, 3.35, 3.60])
T_exp = exp_days / BD


def piecewise_constant_curve(T, F, eps=1e-9):
    """mu constant between consecutive expiries, mu = 0 before the first expiry (S0 = F_1)
    and after the last one (Appendix A, Remark A.1). Represented by doubled knots."""
    T = np.asarray(T, float); F = np.asarray(F, float)
    mus = np.log(F[1:] / F[:-1]) / np.diff(T)
    tk = [0.0, T[0]]; mk = [0.0, 0.0]
    for k, m in enumerate(mus):
        tk += [T[k] + eps, T[k + 1]]; mk += [m, m]
    tk += [T[-1] + eps, T[-1] + 1.0]; mk += [0.0, 0.0]
    return Curve(np.array(tk), np.array(mk)), F[0]


curve, S0 = piecewise_constant_curve(T_exp, fut_px)
r = 0.04

# fixings: every business day d = 1..252 on the first-line contract
fix_days = np.arange(1, BD + 1)
front = np.searchsorted(exp_days, fix_days, side='left')          # index of contract with expiry >= d
tmon = fix_days / BD
Tj = T_exp[front]
n = len(tmon); theta = 1.0 / n
lam = theta * np.exp(curve.M(Tj) - curve.M(tmon))
T_pay = (BD + 5) / BD

# ---- seasonal ATM volatility in calendar time and SSVI surface
a_vol, b_vol, t_peak = 0.48, 0.14, 0.29          # sigma_ATM(t) in [0.34, 0.62], peak in January


def sig_atm(t):
    return a_vol + b_vol * np.cos(2 * np.pi * (np.asarray(t) - t_peak))


_tg = np.linspace(0, 1.2, 12001)
_th = np.concatenate([[0.0], np.cumsum(0.5 * (sig_atm(_tg[1:]) ** 2 + sig_atm(_tg[:-1]) ** 2) * np.diff(_tg))])


def theta_atm(t):                                   # ATM total implied variance
    return np.interp(t, _tg, _th)


rho_s, eta_s, gam_s = 0.25, 0.9, 0.45               # call skew; eta (1+|rho|) = 1.125 <= 2


def phi(th):
    return eta_s * th ** (-gam_s) * (1 + th) ** (gam_s - 1)


def w_ssvi(k, th):
    p = phi(th)
    return 0.5 * th * (1 + rho_s * p * k + np.sqrt((p * k + rho_s) ** 2 + 1 - rho_s ** 2))


def implied_vol(k, t):
    th = theta_atm(t)
    return np.sqrt(w_ssvi(k, th) / t)


def dupire_lv2(k, t, hk=1e-4):
    """sigma_LV^2(k,t) = (dw/dt) / g(k), k = log(K/F(t)); w depends on t only through theta(t)."""
    th = theta_atm(t); dth = sig_atm(t) ** 2
    ht = 1e-6 * th
    wt = (w_ssvi(k, th + ht) - w_ssvi(k, th - ht)) / (2 * ht) * dth
    w = w_ssvi(k, th)
    wk = (w_ssvi(k + hk, th) - w_ssvi(k - hk, th)) / (2 * hk)
    wkk = (w_ssvi(k + hk, th) - 2 * w + w_ssvi(k - hk, th)) / hk ** 2
    g = (1 - k * wk / (2 * w)) ** 2 - wk ** 2 / 4 * (1 / w + 0.25) + wkk / 2
    return wt / g


# tabulate on a daily time grid (row d holds the value at the mid-day t=(d+1/2)/252) and a fine
# moneyness grid; the same tabulated function is used by the PDE and by Monte Carlo
LV_MIN, LV_MAX = 0.10, 1.50
kgrid = np.linspace(-4.0, 4.0, 1601)
nrow = BD + 30
LVtab = np.empty((nrow, kgrid.size))
for d in range(nrow):
    LVtab[d] = np.clip(np.sqrt(np.maximum(dupire_lv2(kgrid, (d + 0.5) / BD), 0.0)), LV_MIN, LV_MAX)


def sigma_lv(t, S):
    d = min(max(int(t * BD), 0), nrow - 1)
    Fs = S0 * np.exp(curve.M(t))
    return np.interp(np.log(np.maximum(S, 1e-12) / Fs), kgrid, LVtab[d])
