"""Competing methods for the comparison study (Section 5.4).

curran          : Curran (1994) conditioning on the geometric average (GBM), general weights and curve
levy_mm         : Levy (1992) / Turnbull-Wakeman two-moment lognormal matching; per-fixing vols allowed
                  (sig_i = constant -> GBM; sig_i = implied vol at K -> 'smile-adjusted' market practice)
implied_vols    : Black implied vols of the vanilla on S_{t_i} at strike K, priced by the PDE with one
                  fixing (exact before the first fixing, Corollary 2.17)
andreasen_pde   : classical change-of-numeraire PDE in x=(A_t-K)/S_t with jump conditions
                  u(t_i^-,x)=u(t_i^+,x+lambda_i) handled by cubic-spline interpolation; same grid,
                  Rannacher and time stepping as pde_price, so the comparison isolates the effect of
                  the continuous variable Y = X - Lambda (Proposition 2.7).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from apo import pde_price, g_known, bs_call_lognormal


def _lognormal_stats(tmon, lam, S0, curve, sig):
    tmon = np.asarray(tmon, float)
    m = np.log(S0) + curve.M(tmon) - 0.5 * sig ** 2 * tmon          # mean of log S_i
    C = sig ** 2 * np.minimum.outer(tmon, tmon)                     # cov of log S_i
    return m, C


def curran(tmon, lam, K, S0, T, r, curve, sig):
    lam = np.asarray(lam, float); Lt = lam.sum(); w = lam / Lt
    m, C = _lognormal_stats(tmon, lam, S0, curve, sig)
    mG = w @ m; vG = w @ C @ w; c = C @ w                           # cov(log S_i, log G)
    v = np.diag(C)
    def condA(g):                                                   # E[A | log G = g] - K
        return np.sum(lam * np.exp(m + c * (g - mG) / vG + 0.5 * (v - c ** 2 / vG))) - K
    lo, hi = mG - 40 * np.sqrt(vG), mG + 40 * np.sqrt(vG)
    if condA(lo) > 0:                                               # deep ITM: exercised a.s.
        return np.exp(-r * T) * (np.sum(lam * np.exp(m + v / 2)) - K)
    g = brentq(condA, lo, hi, xtol=1e-14)
    sG = np.sqrt(vG)
    val = np.sum(lam * np.exp(m + v / 2) * norm.cdf((mG + c - g) / sG)) - K * norm.cdf((mG - g) / sG)
    return np.exp(-r * T) * val


def levy_mm(tmon, lam, K, S0, T, r, curve, sig):
    """Two-moment lognormal matching; sig scalar or array (one vol per fixing).
    E[S_i S_j] = F_i F_j exp(s_{min}^2 t_min), using the vol of the earlier fixing."""
    tmon = np.asarray(tmon, float); lam = np.asarray(lam, float)
    s = np.broadcast_to(np.asarray(sig, float), tmon.shape)
    F = S0 * np.exp(curve.M(tmon))
    i = np.arange(len(tmon)); early = np.minimum.outer(i, i)        # tmon increasing
    cov = np.exp(s[early] ** 2 * tmon[early])
    M1 = lam @ F; M2 = (lam * F) @ cov @ (lam * F)
    v = np.log(M2 / M1 ** 2)
    return bs_call_lognormal(M1, K, v, np.exp(-r * T))


def black_iv(price, F, K, t, df):
    f = lambda s: bs_call_lognormal(F, K, s * s * t, df) - price
    return brentq(f, 1e-4, 5.0, xtol=1e-12)


def implied_vols(tmon, K, S0, r, curve, sigma, sig_ref, Ny=2000):
    out = []
    for t in np.asarray(tmon, float):
        p = pde_price(np.array([t]), np.array([1.0]), K, S0, t, r, curve, sigma, Ny=Ny,
                      dt_max=1 / Ny, Nmin=25, sig_ref=sig_ref)
        out.append(black_iv(p, S0 * np.exp(curve.M(t)), K, t, np.exp(-r * t)))
    return np.array(out)


def andreasen_pde(tmon, lam, K, S0, T, r, curve, sigma, Ny=1000, dt_max=1/252, Nmin=4,
                  width=6.0, sig_ref=None, rannacher=2):
    tmon = np.asarray(tmon, float); lam = np.asarray(lam, float); n = len(tmon)
    xstar = -K / S0
    sr = sig_ref if sig_ref is not None else float(sigma(0.0, np.array([S0]))[0])
    x0 = xstar * np.exp(width * sr * np.sqrt(tmon[-1]))
    jst = int(round(Ny * (1 - xstar / x0))); x0 = xstar / (1 - jst / Ny)
    x = np.linspace(x0, 0.0, Ny + 1); dx = x[1] - x[0]; xi = x[1:-1]
    Sproxy = lambda Lc: -K / (xi - Lc)                              # spot proxy, S = -K/y, y = x - Lambda
    Lam = np.concatenate([[0.0], np.cumsum(lam)])
    tgrid = np.concatenate([[0.0], tmon])
    # value just after the last fixing: e^{-r(T-t_n)} x^+  (zero on x<=0)
    u = np.zeros(Ny + 1)
    for i in range(n, 0, -1):
        ti = tgrid[i]
        # jump condition at t_i: u(t_i^-, x) = u(t_i^+, x + lam_i)
        xs = x + lam[i - 1]
        if i == n:
            new = np.exp(-r * (T - ti)) * np.maximum(xs, 0.0)
        else:
            cs = CubicSpline(x, u)
            inside = xs <= 0.0
            new = np.where(inside, cs(np.minimum(xs, 0.0)), 0.0)
            pos = ~inside
            new[pos] = [g_known(ti, z, tmon, lam, r, T, curve) for z in xs[pos]]
        u = new
        ta = tgrid[i - 1]
        if ti - ta <= 1e-14: continue
        Nt = max(Nmin, int(np.ceil((ti - ta) / dt_max))); h = (ti - ta) / Nt
        steps = []
        for k in range(Nt - 1, -1, -1):
            if k == Nt - 1 and rannacher > 0:
                for mm in range(rannacher):
                    hh = h / rannacher; steps.append((ta + k * h + (rannacher - mm - 1) * hh, hh, 1.0))
            else:
                steps.append((ta + k * h, h, 0.5))
        Lc = Lam[i - 1]
        for (t0, hh, th) in steps:
            tm = t0 + 0.5 * hh if th == 0.5 else t0
            mu = curve.mu(tm); q = r - mu
            w = -mu * xi
            v = 0.5 * sigma(tm, Sproxy(Lc)) ** 2 * xi ** 2
            a = v / dx ** 2 - w / (2 * dx); c = v / dx ** 2 + w / (2 * dx); b = -2 * v / dx ** 2 - q
            bad = np.abs(w) * dx > 2 * v
            if bad.any():
                wp = np.maximum(w, 0); wm = np.minimum(w, 0)
                a = np.where(bad, v / dx ** 2 - wm / dx, a); c = np.where(bad, v / dx ** 2 + wp / dx, c)
                b = np.where(bad, -2 * v / dx ** 2 - (wp - wm) / dx - q, b)
            ab = np.zeros((3, Ny + 1)); ab[1, :] = 1.0
            ab[1, 1:-1] = 1 / hh - th * b; ab[0, 2:] = -th * c; ab[2, :-2] = -th * a
            rhs = u.copy()
            rhs[1:-1] = u[1:-1] / hh + (1 - th) * (a * u[:-2] + b * u[1:-1] + c * u[2:])
            rhs[0] = 0.0; rhs[-1] = g_known(t0, 0.0, tmon, lam, r, T, curve)
            u = solve_banded((1, 1), ab, rhs)
    return S0 * float(CubicSpline(x, u)(xstar))
