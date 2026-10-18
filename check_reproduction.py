"""Compare freshly computed res_*.json with reference_results/ (the values in the paper), and check
that the tables written inline in paper/AsianPDE_V6.tex match the regenerated ones.

CPU times are machine dependent and are skipped (keys starting with 'cpu', 'c1', 'c2', and the third
entry of each row of res_e2.json). res_e10time.json (Table 12) is timing only and is not compared.
"""
import json, glob, os, sys
TOL_REL, TOL_ABS = 1e-8, 1e-12
SKIP_FILES = {'res_e10time.json'}


def is_cpu(key):
    return isinstance(key, str) and (key.startswith('cpu') or key in ('c1', 'c2'))


def walk(a, b, path, out, fname):
    if isinstance(a, dict):
        for k in a:
            if is_cpu(k): continue
            if k not in b: out.append((path + [k], 'missing')); continue
            walk(a[k], b[k], path + [k], out, fname)
    elif isinstance(a, list):
        if len(a) != len(b): out.append((path, 'length')); return
        for i, (x, y) in enumerate(zip(a, b)):
            if fname == 'res_e2.json' and len(path) == 2 and i == 2: continue   # (Ny, price, cpu)
            walk(x, y, path + [i], out, fname)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(a - b)
        if d > TOL_ABS and d > TOL_REL * abs(b): out.append((path, f'{a!r} vs {b!r}'))
    elif a != b:
        out.append((path, f'{a!r} vs {b!r}'))


bad = 0
for ref in sorted(glob.glob('reference_results/res_*.json')):
    f = os.path.basename(ref)
    if f in SKIP_FILES: continue
    if not os.path.exists(f): print(f'{f:28s} NOT PRODUCED'); bad += 1; continue
    out = []; walk(json.load(open(f)), json.load(open(ref)), [], out, f)
    print(f'{f:28s} ' + ('identical' if not out else f'{len(out)} differences, e.g. {out[0]}'))
    bad += bool(out)

tex = open('paper/AsianPDE_V6.tex').read()
for t in ['tab_compare', 'tab_strip', 'tab_greeks']:
    body = open(f'paper/{t}.tex').read().split('\\midrule\n', 1)[1].rsplit('\n\\bottomrule', 1)[0]
    if t == 'tab_greeks':
        for a in ['Two-futures APO, CEV, spot proxy', 'Two-futures APO, CEV, bridge proxy',
                  'Calendar strip, spot proxy', 'Calendar strip, bridge proxy']:
            body = body.replace(a, a.replace(' proxy', ''))
    ok = all(line in tex for line in body.splitlines())
    print(f'{t:28s} ' + ('matches the paper' if ok else 'DIFFERS from the paper')); bad += not ok
print('\nALL CHECKS PASSED' if not bad else f'\n{bad} CHECK(S) FAILED'); sys.exit(bool(bad))
