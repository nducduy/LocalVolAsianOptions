"""Tables for the new Sections 5.4-5.7 (written to paper/tab_*.tex and pasted inline in the paper)."""
import json, glob, numpy as np
L = lambda n: json.load(open(f'res_{n}.json'))
P = 'paper/'
num = {}

# ---------------------------------------------------------------- comparison table
E1 = L('e1'); G = L('e7gbm'); E4 = L('e4'); C = L('e7cev'); E3 = L('e3'); A5 = L('e5apo')
M5 = {(x['sa'], x['beta'], x['K']): x for f in ['e5_0.2', 'e5_0.4'] for x in L(f)}
Gq = {(g['sig'], g['K']): g for g in G['asian']}


def cell(v, ref):
    e = np.abs(np.array(v) - np.array(ref)); rel = 100 * np.max(e / np.array(ref))
    a = f"{e.max():.4f}" if e.max() >= 5e-5 else "$<10^{-4}$"
    r_ = f"{rel:.2f}\\%" if rel >= 0.005 else "$<0.01$\\%"
    return f"{a} ({r_})"


mcA = [d['mc'] for d in E1]; mcB = [d['mc'] for d in E4['gbm']]
mcC = [M5[(d['sa'], d['beta'], d['K'])]['mc'] for d in E3]; mcD = [m['mc'] for m in A5]
rows = [
    ('Constant vol.\\ $\\sigma_{\\rm ATM}$ (PDE)', None, None, [d['gbm'] for d in E3], [d['gbm'] for d in E4['cev']]),
    ('Vorst', [d['vorst'] for d in E1], [d['vorst'] for d in E4['gbm']], None, None),
    ('Levy, ATM vol', [g['levy'] for g in G['asian']], [g['levy'] for g in G['apo']], [c['mm_atm'] for c in C['asian']], [g['levy'] for g in G['apo']]),
    ('Levy, smile vols', None, None, [c['mm_smile'] for c in C['asian']], [c['mm_smile'] for c in C['apo']]),
    ('Curran, ATM vol', [g['curran'] for g in G['asian']], [g['curran'] for g in G['apo']], [Gq[(d['sa'], d['K'])]['curran'] for d in E3], [g['curran'] for g in G['apo']]),
    ('Curran, smile vol', None, None, [c['curran_smile'] for c in C['asian']], [c['curran_smile'] for c in C['apo']]),
    ('PDE, spot proxy', [d['pde'] for d in E1], [d['pde'] for d in E4['gbm']], [d['spot'] for d in E3], [d['spot'] for d in E4['cev']]),
    ('PDE, bridge proxy', [d['pde'] for d in E1], [d['pde'] for d in E4['gbm']], [d['bridge'] for d in E3], [d['bridge'] for d in E4['cev']]),
]
lines = []
for lab, a, b, c, d in rows:
    cs = [cell(v, ref) if v is not None else '--' for v, ref in zip([a, b, c, d], [mcA, mcB, mcC, mcD])]
    lines.append(lab + ' & ' + ' & '.join(cs) + '\\\\')
    if lab.startswith('Curran, smile'): lines.append('\\midrule')
se = [max(d['se'] for d in E1), max(d['se'] for d in E4['gbm']), max(m['se'] for m in M5.values()), max(m['se'] for m in A5)]
fse = lambda s: f'{s:.4f}' if s >= 5e-5 else f'${s:.0e}'.replace('e-0', '\\times10^{-') + '}$'
lines.append('\\midrule Largest MC standard error & ' + ' & '.join(fse(s) for s in se) + '\\\\')
open(P + 'tab_compare.tex', 'w').write(
    "\\begin{tabular}{l llll}\\toprule\n& \\multicolumn{2}{c}{Constant volatility} & \\multicolumn{2}{c}{CEV}\\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nMethod & Asian (12) & APO (5) & Asian (18) & APO (5)\\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- calendar strip
R4 = [L('e8mc4')] + [L(f'e8mc4_{s}') for s in [201, 202, 203]]
R8 = [L(n[4:-5]) for n in sorted(glob.glob('res_e8mc8*.json'))]
pool = lambda R: (np.mean([x['mc'] for x in R], 0), np.sqrt(np.sum(np.square([x['se'] for x in R]), 0)) / len(R))
m4, s4 = pool(R4); m8, s8 = pool(R8)
num['strip_n8'] = len(R8); num['strip_bias'] = list(m8 - m4); num['strip_bias_se'] = list(np.sqrt(s4 ** 2 + s8 ** 2))
mref, sref = m8, s8                                  # finer MC step as the reference
Pr = L('e8price'); lines = []
for p, m, s in zip(Pr, mref, sref):
    e = lambda k: f"{p[k]:.4f} & {p[k] - m:+.4f}"
    lines.append(f"{p['K']:.2f} & {m:.5f} & {s:.5f} & {e('spot')} & {e('bridge')} & {e('mm_smile')} & {e('curran_smile')} & {e('mm_atm')}\\\\")
open(P + 'tab_strip.tex', 'w').write(
    "\\begin{tabular}{c rr rr rr rr rr rr}\\toprule\n& \\multicolumn{2}{c}{Monte Carlo} & \\multicolumn{2}{c}{PDE, spot} & \\multicolumn{2}{c}{PDE, bridge}"
    " & \\multicolumn{2}{c}{Levy, smile} & \\multicolumn{2}{c}{Curran, smile} & \\multicolumn{2}{c}{Levy, ATM}\\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\n"
    "$K$ & Value & SE & Value & Error & Value & Error & Value & Error & Value & Error & Value & Error\\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
num['strip_err'] = {k: [p[k] - m for p, m in zip(Pr, mref)] for k in ['spot', 'bridge', 'mm_smile', 'curran_smile', 'mm_atm']}
num['strip_rel'] = {k: [100 * (p[k] - m) / m for p, m in zip(Pr, mref)] for k in ['spot', 'bridge', 'mm_smile', 'curran_smile', 'mm_atm']}
num['strip_mc'] = list(mref); num['strip_se'] = list(sref)

# ---------------------------------------------------------------- Greeks
GB = L('e9brent'); GS = L('e9strip'); lines = []
f6 = lambda x: f"{x:.5f}"
for d in GB:
    (_, Dg, Gg), (_, Db, Gb), mc = d['grid'], d['bump'], d['mc']
    lab = 'Two-futures APO, ' + ('GBM' if d['model'] == 'GBM' else f"CEV, {d['proxy']} proxy")
    lines.append(f"{lab} & {f6(Dg)} & {f6(Db)} & {f6(mc[1][0])} ({mc[1][1]:.0e}) & {f6(Gg)} & {f6(Gb)} & {f6(mc[2][0])} ({mc[2][1]:.0e})\\\\")
mcS = GS[-1]['mc']
for d in GS[:-1]:
    (_, Dg, Gg), (_, Db, Gb) = d['grid'], d['bump']
    lines.append(f"Calendar strip, {d['proxy']} proxy & {f6(Dg)} & {f6(Db)} & {f6(mcS[1][0])} ({mcS[1][1]:.0e}) & {f6(Gg)} & {f6(Gb)} & {f6(mcS[2][0])} ({mcS[2][1]:.0e})\\\\")
open(P + 'tab_greeks.tex', 'w').write(
    "\\begin{tabular}{l rrr rrr}\\toprule\n& \\multicolumn{3}{c}{Delta} & \\multicolumn{3}{c}{Gamma}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
    "Case & Grid & Bump & MC (SE) & Grid & Bump & MC (SE)\\\\\\midrule\n" + "\n".join(lines).replace('e-0', 'e-') + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- CPU
T = L('e10time'); lines = []; prev = None
for d in T['cases']:
    nm = {'Asian, 12 monthly': 'Asian, 12 monthly fixings', 'Brent APO, 21 daily': 'Two-futures APO, 21 daily fixings',
          'Calendar strip, 252 daily': 'Calendar strip, 252 daily fixings'}[d['product']]
    prod = nm if d['product'] != prev else ''; prev = d['product']
    lines.append(f"{prod} & {d['model']} & {d['steps1000']} & {d['cpu1000']:.2f} & {d['cpu2000']:.2f}\\\\")
open(P + 'tab_cpu.tex', 'w').write(
    "\\begin{tabular}{ll rrr}\\toprule\n& & Time steps & \\multicolumn{2}{c}{CPU per price (s)}\\\\\n\\cmidrule(lr){4-5}\n"
    "Product & Model / proxy & ($N_y=1000$) & $N_y=1000$ & $N_y=2000$\\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n")
num['time'] = T
json.dump(num, open('res_numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in num.items() if k != 'time'}, indent=1, default=float))
