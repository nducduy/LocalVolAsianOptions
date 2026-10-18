"""Reproduce every table and figure of the paper.

    python run_all.py                 # run all experiments, then build tables, figures and the PDF
    python run_all.py --from-reference # skip the experiments: copy reference_results/ and rebuild
    python run_all.py --only e8        # run only the steps whose name starts with 'e8'

Results are written as res_*.json in this folder; tables, figures and macros.tex go to paper/.
The full run takes about two to three hours on one core (Monte Carlo dominates).
Run timing.py on an otherwise idle machine; its numbers (Table 12) depend on the hardware.
"""
import subprocess, sys, shutil, glob, os, time
STEPS = [
    # name, command                                   # produces / used in
    ('e1', 'e1.py'),                                   # Table 1
    ('e2', 'e2.py'),                                   # Table 2, Figure 1
    ('e3', 'e3.py'),                                   # Table 5 (PDE prices)
    ('e4', 'e4.py'),                                   # Tables 4, 6, 7
    ('e5 prog', 'e5.py prog'),                         # Table 7 (CEV, fine-step MC)
    ('e5 apo', 'e5.py apo'),                           # Table 6 (MC)
    ('e5 0.2', 'e5.py 0.2'), ('e5 0.4', 'e5.py 0.4'),  # Table 5 (MC)
    ('e6', 'e6.py'),                                   # Section 3.3 parity check, Remark 2.3, Figure 3
    ('e7 wp', 'e7.py wp'),                             # Figure 4
    ('e7 gbm', 'e7.py gbm'), ('e7 cev', 'e7.py cev'),  # Table 8
    ('e7 curransmile', 'e7.py curransmile'), ('e7 apocontract', 'e7.py apocontract'),
    ('e7 mctime', 'e7.py mctime'),                     # Section 5.4 (MC cost)
    ('e8 calib', 'e8.py calib'),                       # Figure 5(c)
    ('e8 conv', 'e8.py conv'),                         # Section 5.5 (convergence)
    ('e8 price', 'e8.py price'),                       # Table 10
] + [(f'e8 {p}', f'e8.py {p}') for p in ['mc4', 'mc4_201', 'mc4_202', 'mc4_203',
                                          'mc8', 'mc8_301', 'mc8_302', 'mc8_303']] + [
    ('greeks brent', 'greeks.py brent'), ('greeks strip', 'greeks.py strip'),  # Table 11
    ('timing', 'timing.py'),                           # Table 12
    ('text_numbers', 'text_numbers.py'),               # numbers quoted in the text (not in tables)
]
BUILD = [('fig_strike', 'fig_strike.py'),              # Figure 2
         ('mk', 'mk.py'),                              # Tables 1-7, Figures 1 and 3, macros.tex
         ('mk2', 'mk2.py'),                            # Tables 8, 10-12 (inline in the paper)
         ('fig_wp', 'fig_wp.py'), ('fig_market', 'fig_market.py')]


def run(name, cmd):
    t0 = time.time(); print(f'== {name}', flush=True)
    subprocess.run([sys.executable] + cmd.split(), check=True)
    print(f'   done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    os.makedirs('paper', exist_ok=True)
    args = sys.argv[1:]
    if '--only' in args:
        pre = args[args.index('--only') + 1]
        for n, c in STEPS + BUILD:
            if n.startswith(pre): run(n, c)
        sys.exit()
    if '--from-reference' in args:
        for f in glob.glob('reference_results/res_*.json'): shutil.copy(f, '.')
    else:
        for n, c in STEPS: run(n, c)
    for n, c in BUILD: run(n, c)
    if shutil.which('pdflatex'):
        for _ in range(3):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', 'AsianPDE_V6.tex'], cwd='paper',
                           stdout=subprocess.DEVNULL)
        print('paper/AsianPDE_V6.pdf built')
