import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.stats import norm

# ---------------------------------------------------------------- curve
class Curve:
    """Piecewise-linear mu(t) on knots tk (tk[0]=0), constant mu_k[-1] beyond last knot.
    M(t) = int_0^t mu(s) ds (exact)."""
    def __init__(self, tk, muk):
        self.tk = np.asarray(tk, float); self.muk = np.asarray(muk, float)
        seg = np.diff(self.tk) * 0.5 * (self.muk[1:] + self.muk[:-1])
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
    @classmethod
    def flat(cls, mu):
        return cls([0.0, 1.0], [mu, mu])
    def mu(self, t):
        return np.interp(t, self.tk, self.muk)
    def M(self, t):
        t = np.atleast_1d(np.asarray(t, float)); out = np.empty_like(t)
        for n, s in enumerate(t):
            if s >= self.tk[-1]:
                out[n] = self.cum[-1] + self.muk[-1] * (s - self.tk[-1]); continue
            k = np.searchsorted(self.tk, s, side='right') - 1
            mus = self.mu(s)
            out[n] = self.cum[k] + (s - self.tk[k]) * 0.5 * (self.muk[k] + mus)
        return out if out.size > 1 else out[0]

def fit_curve(T, F):
    """Appendix A: piecewise-linear mu matching futures F at maturities T, flat-left
    (F_0=F_1 at T_0=0) and mu=0 at the last knot. Returns (Curve, S0)."""
    T = np.concatenate([[0.0], np.asarray(T, float)]); F = np.concatenate([[F[0]], F])
    N = len(T) - 1; mu = np.zeros(N + 1)
    for k in range(N - 1, -1, -1):
        d = T[k + 1] - T[k]
        mu[k] = -mu[k + 1] + 2.0 / d * np.log(F[k + 1] / F[k])
    c = Curve(T, mu); c.muk_end = 0.0
    return c, F[0]

# ---------------------------------------------------------------- PDE
def g_known(t, x, tmon, lam, r, T, curve):
    """value f(t,x) for x>=0 (eq. (14)); sums over t_j > t."""
    fut = tmon > t + 1e-14
    q = r * (tmon[fut] - t) - (curve.M(tmon[fut]) - curve.M(t)) if fut.any() else 0.0
    s = np.sum(lam[fut] * np.exp(-r * (T - tmon[fut]) - q)) if fut.any() else 0.0
    return np.exp(-r * (T - t)) * x + s

def pde_price(tmon, lam, K, S0, T, r, curve, sigma, Ny=1000, dt_max=1/252, Nmin=4,
              y0=None, width=6.0, sig_ref=None, rannacher=2, return_grid=False, proxy='spot', put=False, bridge_freeze=False):
    """Fixed-grid Crank-Nicolson solution of (19) for a call on A_T = sum lam_i S_{t_i}.
    sigma(t, S) local vol (vectorised in S). Proxy S = -K/y.
    bridge_freeze=True evaluates the bridge proxy once per monitoring interval (at its midpoint)
    instead of at every time step; intended for daily fixings."""
    tmon = np.asarray(tmon, float); lam = np.asarray(lam, float)
    n = len(tmon); Lam = np.concatenate([[0.0], np.cumsum(lam)])     # Lam[i] = Lambda_{t_i}
    ystar = -K / S0
    if y0 is None:
        sr = sig_ref if sig_ref is not None else float(sigma(0.0, np.array([S0]))[0])
        y0 = ystar * np.exp(width * sr * np.sqrt(tmon[-1]))
    jst = int(round(Ny * (1 - ystar / y0)))           # put y* exactly on a node
    y0 = ystar / (1 - jst / Ny)
    y = np.linspace(y0, 0.0, Ny + 1); dy = y[1] - y[0]
    yi = y[1:-1]; Sproxy = -K / yi
    # terminal condition at t_n
    # cell-averaged terminal payoff (removes grid-position dependence of the kink)
    lo = y - 0.5 * dy + Lam[n]; hi = y + 0.5 * dy + Lam[n]
    P = lambda z: 0.5 * np.maximum(z, 0.0)**2
    if put:
        Pm = lambda z: 0.5 * np.maximum(-z, 0.0)**2
        u = np.exp(-r * (T - tmon[-1])) * (Pm(lo) - Pm(hi)) / dy
        u[-1] = 0.0
        u[0] = -g_known(tmon[-1], y0 + Lam[n], tmon, lam, r, T, curve)
    else:
        u = np.exp(-r * (T - tmon[-1])) * (P(hi) - P(lo)) / dy
        u[-1] = np.exp(-r * (T - tmon[-1])) * Lam[n]
    tgrid = np.concatenate([[0.0], tmon])
    for i in range(n, 0, -1):
        ta, tb = tgrid[i - 1], tgrid[i]
        if tb - ta <= 1e-14: continue
        L = Lam[i - 1]; x = yi + L
        Nt = max(Nmin, int(np.ceil((tb - ta) / dt_max)))
        h = (tb - ta) / Nt
        steps = []
        for k in range(Nt - 1, -1, -1):
            if k == Nt - 1 and rannacher > 0:
                for m in range(rannacher):  # implicit Euler sub-steps
                    hh = h / rannacher
                    steps.append((ta + k * h + (rannacher - m - 1) * hh, hh, 1.0))
            else:
                steps.append((ta + k * h, h, 0.5))
        Spf = None
        if proxy == 'bridge' and i > 1 and bridge_freeze:
            Spf = bridge_spot(x, 0.5 * (ta + tb), tmon[:i - 1], lam[:i - 1], K, S0, curve)
        for (t0, hh, th) in steps:
            tm = t0 + 0.5 * hh if th == 0.5 else t0
            mu = curve.mu(tm); q = r - mu
            w = -mu * x
            if Spf is not None:
                Sp = Spf
            elif proxy == 'bridge' and i > 1:
                Sp = bridge_spot(x, tm, tmon[:i - 1], lam[:i - 1], K, S0, curve)
            else:
                Sp = Sproxy
            v = 0.5 * sigma(tm, Sp) ** 2 * x ** 2
            a = v / dy**2 - w / (2 * dy); c = v / dy**2 + w / (2 * dy); b = -2 * v / dy**2 - q
            bad = np.abs(w) * dy > 2 * v                       # Peclet violated -> upwind
            if bad.any():
                wp = np.maximum(w, 0); wm = np.minimum(w, 0)
                a = np.where(bad, v / dy**2 - wm / dy, a)
                c = np.where(bad, v / dy**2 + wp / dy, c)
                b = np.where(bad, -2 * v / dy**2 - (wp - wm) / dy - q, b)
            ab = np.zeros((3, Ny + 1))
            ab[1, :] = 1.0
            ab[1, 1:-1] = 1 / hh - th * b
            ab[0, 2:] = -th * c                                 # super-diag
            ab[2, :-2] = -th * a                                # sub-diag
            rhs = u.copy()
            rhs[1:-1] = u[1:-1] / hh + (1 - th) * (a * u[:-2] + b * u[1:-1] + c * u[2:])
            rhs[0] = -g_known(t0, y0 + L, tmon, lam, r, T, curve) if put else 0.0
            rhs[-1] = 0.0 if put else g_known(t0, L, tmon, lam, r, T, curve)     # y=0 -> x = Lambda >= 0
            u = solve_banded((1, 1), ab, rhs)
    cs = CubicSpline(y, u)
    price = S0 * float(cs(ystar))
    if return_grid: return price, y, u
    return price

def bridge_spot(x, t, tpast, lpast, K, S0, curve, iters=12):
    """Bridge proxy: solve sum_j l_j c_j S^{rho_j} - K = x S (x<0) for S, where
    S_{t_j} ~ S0^{1-rho_j} S_t^{rho_j} e^{M(t_j)-rho_j M(t)}, rho_j = t_j/t.
    For x >= 0 (vol irrelevant) return the spot proxy."""
    rho = tpast / t
    c = lpast * S0 ** (1 - rho) * np.exp(curve.M(tpast) - rho * curve.M(t))
    neg = x < 0
    xs = np.where(neg, x, -1.0)
    S = K / (-xs)                                    # F(S) >= 0 here; Newton from the right
    for _ in range(iters):
        Sr = S[:, None] ** rho[None, :]
        F = Sr @ c - K - xs * S
        dF = (Sr / S[:, None]) @ (c * rho) - xs
        S = np.maximum(S - F / dF, 1e-12)
    return np.where(neg, S, K / np.maximum(-x, 1e-300))

# ---------------------------------------------------------------- closed forms (GBM)
def geo_stats(tmon, lam, S0, curve, sig):
    tmon = np.asarray(tmon); lam = np.asarray(lam); Lt = lam.sum(); wgt = lam / Lt
    m = np.sum(wgt * (np.log(S0) + curve.M(tmon) - 0.5 * sig**2 * tmon))
    v = sig**2 * np.sum(np.outer(wgt, wgt) * np.minimum.outer(tmon, tmon))
    return Lt, m, v

def bs_call_lognormal(Fwd, K, v, df):
    if K <= 0: return df * (Fwd - K)
    d1 = (np.log(Fwd / K) + 0.5 * v) / np.sqrt(v); d2 = d1 - np.sqrt(v)
    return df * (Fwd * norm.cdf(d1) - K * norm.cdf(d2))

def geo_call(tmon, lam, K, S0, T, r, curve, sig):
    Lt, m, v = geo_stats(tmon, lam, S0, curve, sig)
    return bs_call_lognormal(Lt * np.exp(m + 0.5 * v), K, v, np.exp(-r * T))

def vorst(tmon, lam, K, S0, T, r, curve, sig):
    Lt, m, v = geo_stats(tmon, lam, S0, curve, sig)
    EA = np.sum(np.asarray(lam) * S0 * np.exp(curve.M(np.asarray(tmon))))
    EG = Lt * np.exp(m + 0.5 * v)
    return bs_call_lognormal(EG, K - (EA - EG), v, np.exp(-r * T))

def forward_value(tmon, lam, K, S0, T, r, curve):
    """put-call parity: C - P"""
    return np.exp(-r * T) * (np.sum(np.asarray(lam) * S0 * np.exp(curve.M(np.asarray(tmon)))) - K)

# ---------------------------------------------------------------- Monte Carlo
def mc_price(tmon, lam, Ks, S0, T, r, curve, sigma, sig_cv, npaths=10**6, sub_dt=1/252,
             chunk=200_000, seed=1, gbm_exact=False, t_start=0.0, A_past=0.0):
    """Log-Euler MC (antithetic) for local vol, several strikes on common paths, with a
    geometric-average GBM control variate driven by the same normals.
    t_start/A_past: in-progress valuation (monitoring dates > t_start, past fixings A_past)."""
    rng = np.random.default_rng(seed)
    Ks = np.atleast_1d(np.asarray(Ks, float))
    tmon = np.asarray(tmon, float); lam = np.asarray(lam, float)
    tg = np.concatenate([[t_start], tmon])
    grid = [t_start]; is_mon = [False]
    for i in range(1, len(tg)):
        m = 1 if gbm_exact else max(1, int(np.ceil((tg[i] - tg[i - 1]) / sub_dt - 1e-9)))
        for k in range(1, m + 1):
            grid.append(tg[i - 1] + (tg[i] - tg[i - 1]) * k / m); is_mon.append(k == m)
    grid = np.array(grid); is_mon = np.array(is_mon); dts = np.diff(grid)
    dM = np.diff(curve.M(grid))
    Lt = lam.sum(); wgt = lam / Lt
    # CV mean: geometric on remaining fixings with shifted strike, from t_start
    tloc = tmon - t_start
    cur_loc = ShiftCurve(curve, t_start)
    cv_mean = np.array([geo_call(tloc, lam, K - A_past, S0, T - t_start, r, cur_loc, sig_cv) for K in Ks])
    disc = np.exp(-r * (T - t_start))
    Y = [[] for _ in Ks]; C = [[] for _ in Ks]
    for c0 in range(0, npaths, chunk):
        nb = min(chunk, npaths - c0) // 2
        lnS = np.full(2 * nb, np.log(S0)); lnG = np.full(2 * nb, np.log(S0))
        A = np.zeros(2 * nb); LG = np.zeros(2 * nb); j = 0
        for k in range(len(dts)):
            z = rng.standard_normal(nb); z = np.concatenate([z, -z]); dW = np.sqrt(dts[k]) * z
            s = sig_cv if gbm_exact else sigma(grid[k], np.exp(lnS))
            lnS += dM[k] - 0.5 * s**2 * dts[k] + s * dW
            lnG += dM[k] - 0.5 * sig_cv**2 * dts[k] + sig_cv * dW
            if is_mon[k + 1]:
                A += lam[j] * np.exp(lnS); LG += wgt[j] * lnG; j += 1
        G = Lt * np.exp(LG)
        for q_, K in enumerate(Ks):
            y_ = disc * np.maximum(A_past + A - K, 0); c_ = disc * np.maximum(G - (K - A_past), 0)
            Y[q_].append(0.5 * (y_[:nb] + y_[nb:])); C[q_].append(0.5 * (c_[:nb] + c_[nb:]))
    res = []
    for q_ in range(len(Ks)):
        y_ = np.concatenate(Y[q_]); c_ = np.concatenate(C[q_])
        cov = np.cov(y_, c_); b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        z_ = y_ - b * (c_ - cv_mean[q_])
        res.append((z_.mean(), z_.std(ddof=1) / np.sqrt(len(z_))))
    return np.array(res)

class ShiftCurve:
    """curve seen from time t0 (local time s = t - t0)."""
    def __init__(self, c, t0): self.c = c; self.t0 = t0; self.M0 = c.M(t0)
    def mu(self, s): return self.c.mu(np.asarray(s) + self.t0)
    def M(self, s): return self.c.M(np.asarray(s) + self.t0) - self.M0
