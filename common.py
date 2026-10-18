import numpy as np, json, time
from apo import *
def cev(beta, satm, S0):
    sg = satm * S0**(1 - beta)
    return lambda t, S: sg * np.maximum(S, 1e-8)**(beta - 1)
def const(s): return lambda t, S: s * np.ones_like(S)
def save(name, obj): json.dump(obj, open(f'res_{name}.json', 'w'), indent=1)
# Brent-like setup (business-day clock, t = d/252)
r_b = 0.045
fut_days = np.array([11, 32, 53, 74, 95, 116, 137, 158])
fut_px = np.array([82.40, 82.05, 81.62, 81.20, 80.85, 80.42, 80.10, 79.71])
curve_b, S0_b = fit_curve(fut_days / 252, fut_px)
d1 = np.arange(106, 117); d2 = np.arange(117, 127)
theta = 1 / 21
T1, T2 = 116 / 252, 137 / 252
tmon_b = np.concatenate([d1, d2]) / 252
Tj = np.concatenate([np.full(11, T1), np.full(10, T2)])
lam_b = theta * np.exp(curve_b.M(Tj) - curve_b.M(tmon_b))
T_b = 128 / 252
