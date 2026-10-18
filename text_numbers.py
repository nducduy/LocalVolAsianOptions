"""Numbers quoted in the text of the paper that are not in a table or in macros.tex.

Part 1 runs new computations (absorption simulation, Table 1 refinement, frozen bridge proxy, bump
sizes for the strip Greeks). Part 2 derives the remaining quoted numbers from the stored results.
Each number is printed next to the place in the paper that quotes it, and all are saved in
res_textnumbers.json (CPU-time entries start with 'cpu').
"""
import glob, json, time
import numpy as np
from common import *
import realistic as R
from greeks import bump_greeks

L = lambda n: json.load(open(f'res_{n}.json'))
out = {}


def show(key, value, where, quoted):
    if isinstance(value, list): value = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
    out[key] = value
    print(f'{where:22s} {key:34s} {value!s:>28s}   (paper: {quoted})')


r = 0.05; cur = Curve.flat(0.05); S0 = 100; tm = np.arange(1, 13) / 12; lam = np.ones(12) / 12

# ============================================================== Part 1: new computations
# Remark 2.3: Euler simulation with absorption, worst case beta=0.25, sigma_ATM=0.4
beta, sa = 0.25, 0.4; sig = sa * S0 ** (1 - beta); dt = 1 / 1008; Ks = [85, 100, 115]
rng = np.random.default_rng(2024); n_abs = 0; pos = {K: 0 for K in Ks}; npaths = 10 ** 6
for c0 in range(0, npaths, 250_000):
    S = np.full(250_000, float(S0)); A = np.zeros(250_000); dead = np.zeros(250_000, bool); j = 0
    for k in range(1008):
        z = rng.standard_normal(250_000)
        S = np.where(dead, 0.0, S + r * S * dt + sig * np.maximum(S, 0) ** beta * np.sqrt(dt) * z)
        dead |= S <= 0; S = np.where(dead, 0.0, S)
        if (k + 1) % 84 == 0: A += lam[j] * S; j += 1
    n_abs += int(dead.sum())
    for K in Ks: pos[K] += int(np.sum(dead & (A > K)))
show('absorbed_paths', n_abs, 'Remark 2.3', '1,161 absorbed paths (10^6 paths, step 1/1008)')
show('absorbed_paths_positive_payoff', sum(pos.values()), 'Remark 2.3', 'none had a positive payoff')

# Section 5.1.1: deep out-of-the-money, low-volatility case of Table 1 refined to N_y=8000
show('table1_sig0.1_K115_Ny8000', round(pde_price(tm, lam, 115, S0, 1, r, cur, const(0.1), Ny=8000, dt_max=1 / 8000), 6),
     'Section 5.1.1', 'refining to N_y=8000 gives 0.089488')

# Section 5.5: bridge proxy evaluated once per interval vs at every step (two-futures APO, CEV)
sf = cev(0.5, 0.35, S0_b); res = {}
for fr in (False, True):
    t0 = time.perf_counter()
    res[fr] = pde_price(tmon_b, lam_b, 80, S0_b, T_b, r_b, curve_b, sf, Ny=2000, dt_max=1 / 2000, Nmin=25,
                        sig_ref=0.35, proxy='bridge', bridge_freeze=fr)
    res[(fr, 't')] = time.perf_counter() - t0
show('freeze_price_change', float(f'{abs(res[True] - res[False]):.1e}'), 'Section 5.5', '4e-8')
show('cpu_freeze_speedup', round(res[(False, 't')] / res[(True, 't')], 1), 'Section 5.5', 'factor of three')

# Section 5.6: dependence of the bumped strip Greeks on the bump size h
K = 3.38
for proxy in ('spot', 'bridge'):
    kw = dict(Ny=1000, dt_max=1 / 1000, Nmin=8, sig_ref=0.5, proxy=proxy, bridge_freeze=True)
    g = {hr: [round(float(x), 5) for x in bump_greeks(R.tmon, R.lam, K, R.S0, R.T_pay, R.r, R.curve,
                                                      R.sigma_lv, hr * R.S0, **kw)[1:]]
         for hr in (0.005, 0.01, 0.02, 0.04)}
    show(f'strip_bump_{proxy}_(delta,gamma)', g, 'Section 5.6', 'bumped gammas depend on h')

# ============================================================== Part 2: derived from stored results
E1, E4, A5, P5 = L('e1'), L('e4'), L('e5apo'), L('e5prog')
z1 = sorted((abs(d['pde'] - d['mc']) / d['se'], d['sig'], d['K']) for d in E1)
show('table1_max_se_multiple_excl', round(z1[-2][0], 2), 'Section 5.1.1', 'within one or two SE in almost every case')
show('table1_outlier_se_multiple', round(z1[-1][0], 2), 'Section 5.1.1', 'the exception (sigma=0.1, K=115)')
show('table4_max_se_multiple', round(max(abs(d['pde'] - d['mc']) / d['se'] for d in E4['gbm']), 2), 'Section 5.1.2', '1.5 SE')
show('table4_max_abs_error', float(f"{max(abs(d['pde'] - d['mc']) for d in E4['gbm']):.1e}"), 'Section 5.1.2', 'about 4e-5')
dev = [(abs(d[p] - m['mc']) / m['se'], p, d['K']) for d, m in zip(E4['cev'], A5) for p in ('spot', 'bridge')]
show('table6_max_se_multiple', round(max(dev)[0], 2), 'Section 5.2.2', '1.53 SE (bridge, K=88)')
show('table6_spot_closer_at', [d['K'] for d, m in zip(E4['cev'], A5) if abs(d['spot'] - m['mc']) < abs(d['bridge'] - m['mc'])],
     'Section 5.2.2', 'K=72,84,88')
gp = [abs(d['price'] - d['mc']) / d['se'] for d in E4['prog'] if d['model'] == 'gbm']
show('table7_gbm_max_se_multiple', round(max(gp), 2), 'Section 5.3', 'at most 2.6 SE')
show('table7_bridge_max_rel_pct', round(max(abs(100 * (d['bridge'] - d['mc']) / d['mc']) for d in P5), 2), 'Section 5.3', '0.2%')
show('table7_spot_max_rel_pct', round(max(abs(100 * (d['spot'] - d['mc']) / d['mc']) for d in P5), 2), 'Section 5.3', '0.8%')

# Figure 4
W = L('e7wp'); mx = lambda e: max(abs(v) for v in e.values())
show('fig4_richardson_500_1000', float(f"{mx(W['rich'][2]['err']):.1e}"), 'Section 5.4', '1.3e-6')
show('fig4_richardson_1000_2000', float(f"{mx(W['rich'][3]['err']):.1e}"), 'Section 5.4', '1e-7')
A = W['andr']; ra = [max(abs((4 * A[i]['err'][k] - A[i - 1]['err'][k]) / 3) for k in A[i]['err']) for i in (2, 3)]
show('fig4_jump_richardson', [float(f'{x:.1e}') for x in ra], 'Section 5.4', '3.9e-4 and 5.5e-5')
mcse = mx(W['mc'][-1]['se'])
show('fig4_mc_se_2.56e6', float(f'{mcse:.1e}'), 'Section 5.4', '1.8e-4')
show('fig4_mc_paths_factor_for_1e-6', float(f'{(mcse / 1e-6) ** 2:.1e}'), 'Section 5.4', 'about 3e4 times as many paths')
show('fig4_curran_vorst_levy', [float(f"{mx(W['closed'][k]['err']):.1e}") for k in ('curran', 'vorst', 'levy')], 'Section 5.4', '7e-4; 3-6e-2')

# Table 8 ratios and SE multiples
C, E3 = L('e7cev'), L('e3'); M5 = {(x['sa'], x['beta'], x['K']): x for f in ['e5_0.2', 'e5_0.4'] for x in L(f)}
mc = np.array([M5[(d['sa'], d['beta'], d['K'])]['mc'] for d in E3])
err = lambda v: np.abs(np.array(v) - mc)
br = err([d['bridge'] for d in E3]); lv = err([c['mm_smile'] for c in C['asian']]); cu = err([c['curran_smile'] for c in C['asian']])
rms = lambda e: np.sqrt(np.mean(e ** 2))
show('table8_max_ratio_levy_curran', [round(lv.max() / br.max(), 1), round(cu.max() / br.max(), 1)], 'Section 5.4', '6.5 and 4.6')
show('table8_rms_bridge_levy_curran', [round(rms(br), 4), round(rms(lv), 3), round(rms(cu), 3)], 'Section 5.4', '0.0071; 0.054, 0.031')
show('table8_rms_ratio', [round(rms(lv) / rms(br), 1), round(rms(cu) / rms(br), 1)], 'Section 5.4', '7.6 and 4.3')
sm = [abs(x[k] - m['mc']) / m['se'] for x, m in zip(C['apo'], A5) for k in ('mm_smile', 'curran_smile')]
show('apo_cev_smile_se_multiples', [round(min(sm), 1), round(max(sm), 1)], 'Section 5.4', '2.0 to 7.3 SE')

# Section 5.5
cal = L('e8calib'); d = sorted(abs(x['iv'] - x['target']) for x in cal)
show('calib_max_vol_points', round(100 * d[-1], 2), 'Section 5.5', 'within 0.15 vol points')
show('calib_second_largest', round(100 * d[-2], 2), 'Section 5.5', '0.07 in all but one case')
pool = lambda Rr: (np.mean([x['mc'] for x in Rr], 0), np.sqrt(np.sum(np.square([x['se'] for x in Rr]), 0)) / len(Rr))
m4, s4 = pool([L('e8mc4')] + [L(f'e8mc4_{s}') for s in (201, 202, 203)])
m8, s8 = pool([L('e8mc8')] + [L(f'e8mc8_{s}') for s in (301, 302, 303)])
show('strip_mc_step_diff_max', float(f'{np.max(np.abs(m8 - m4)):.1e}'), 'Section 5.5', 'at most 1.2e-4')
show('strip_mc_combined_se_max', float(f'{np.max(np.sqrt(s4 ** 2 + s8 ** 2)):.1e}'), 'Section 5.5', '2.3e-4')
Pr = L('e8price'); be = np.array([p['bridge'] for p in Pr]) - m8
show('strip_bridge_max_abs_err', round(float(np.max(np.abs(be))), 4), 'Section 5.5', '0.0006')
show('strip_bridge_max_rel_pct', round(float(np.max(np.abs(be) / m8) * 100), 2), 'Section 5.5', '0.11%')
show('strip_bridge_max_se_multiple', round(float(np.max(np.abs(be) / s8)), 1), 'Section 5.5', 'at most 3.5 SE')
cv = {(d['proxy'], d['Ny'], d['Nmin'], d['K']): d['price'] for d in L('e8conv')}
show('strip_refinement_max_change', float(f"{max(abs(cv[('bridge', 4000, 25, k)] - [p['bridge'] for p in Pr if p['K'] == k][0]) for k in (2.7, 3.38, 4.06)):.1e}"),
     'Section 5.5', 'at most 6e-5')
rat = [(cv[('spot', 1000, 4, k)] - cv[('spot', 1000, 8, k)]) / (cv[('spot', 1000, 8, k)] - cv[('spot', 1000, 16, k)]) for k in (2.7, 3.38, 4.06)]
show('strip_time_error_ratio', [round(x, 2) for x in rat], 'Section 5.5', 'decreases by a factor of four')

# Section 5.6 (Table 11)
GS = L('e9strip'); mcS = GS[-1]['mc']; sp, bg = GS[0], GS[1]
show('strip_spot_grid_se_multiples', [round(abs(sp['grid'][1] - mcS[1][0]) / mcS[1][1], 1), round(abs(sp['grid'][2] - mcS[2][0]) / mcS[2][1], 1)],
     'Section 5.6', 'within two SE')
show('strip_bridge_grid_delta_bias_pct', round(100 * (bg['grid'][1] / mcS[1][0] - 1), 2), 'Section 5.6', '1.2% too high')
GB = L('e9brent')
show('brent_gamma_se_multiples', [round(abs(d['grid'][2] - d['mc'][2][0]) / d['mc'][2][1], 2) for d in GB], 'Section 5.6', 'within one SE')
dl = [(d['grid'][1], d['bump'][1], d['mc'][1][0]) for d in GB]
show('brent_delta_max_pairwise_diff', float(f'{max(max(t) - min(t) for t in dl):.1e}'), 'Section 5.6', 'deltas agree to 2e-5')

json.dump(out, open('res_textnumbers.json', 'w'), indent=1, default=float)
