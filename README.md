# LocalVolAsianOptions

Code for J. L. Kirkby and D. Nguyen, *Average Price Asian Options: A PDE Method for Local Volatility*.
This folder reproduces every table and figure of the paper (`paper/AsianPDE_V6.tex`).

## Requirements

Python 3.10 or later with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`).
Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. `pdflatex` is optional; it is
only needed to build the PDF.

## Quick start

```
python run_all.py --from-reference   # rebuild all tables, figures and the PDF from the stored results (about 2 minutes)
python run_all.py                    # rerun every experiment from scratch, then rebuild (2-3 hours on one core)
python check_reproduction.py         # compare a fresh run with reference_results/ and with the paper
python run_all.py --only e8          # run a single group of steps
```

All Monte Carlo runs use fixed seeds, so a fresh run reproduces the stored results exactly, apart from
CPU times. Run `timing.py` (Table 12) on an otherwise idle machine; its numbers depend on the hardware.

## Verification

On 18 October 2026, a clean copy of this folder with all results deleted was run from scratch
(all steps of `run_all.py`, in the order listed there; Python 3.13 on Linux). `check_reproduction.py` found every results file identical to
`reference_results/` (relative tolerance 1e-8; CPU times excluded), and the regenerated tables match
those in the paper. Tables 1-7 and `macros.tex` were also regenerated identically, apart from CPU times.
`text_numbers.py` recomputes the numbers quoted in the text outside tables; its run led to three small
corrections in the paper (1,161 absorbed paths in Remark 2.3, an RMS ratio of 7.6 in Section 5.4, and
2.6 standard errors in Section 5.3).

The put-call check in `e6.py` now covers the GBM model and both CEV proxies. The `res_e6.json`
previously stored in the repository was out of date: it appears to have been produced with an earlier
version of the code, and showed CEV residuals of 0.017-0.051, the size of error that Section 3.3
attributes to imposing u=0 at the lower boundary of the put. With the
current code all residuals are below 1.3e-7, as stated in the paper.

## Library modules

| file | contents |
|---|---|
| `apo.py` | PDE solver `pde_price` (Algorithm 1: Crank-Nicolson with Rannacher steps, upwinding, spot and bridge proxies, puts), Monte Carlo `mc_price`, curve fitting (Appendix A), Vorst and geometric closed forms |
| `common.py` | CEV and constant local volatilities, the two-futures (Brent-style) set-up of Section 5.1.2 |
| `bench.py` | competing methods of Section 5.4: Curran, Levy, Black implied volatilities, PDE with jump conditions |
| `realistic.py` | Section 5.5: seasonal futures curve, SSVI surface, Dupire local volatility table |
| `greeks.py` | Section 5.6: grid, bump-and-revalue and Monte Carlo Greeks |

## Experiments: which script produces what

| script | results file(s) | paper |
|---|---|---|
| `e1.py` | `res_e1.json` | Table 1 |
| `e2.py` | `res_e2.json` | Table 2, Figure 1 |
| `fig_strike.py` | `res_figstrike.json`, `paper/fig_conv_strike.pdf` | Figure 2 |
| `e4.py` | `res_e4.json` | Tables 4, 6 (PDE prices), 7 (GBM rows and parity) |
| `e3.py` | `res_e3.json` | Table 5 (PDE prices) |
| `e5.py 0.2`, `e5.py 0.4` | `res_e5_0.2.json`, `res_e5_0.4.json` | Table 5 (Monte Carlo, step 1/1008) |
| `e5.py apo` | `res_e5apo.json` | Table 6 (Monte Carlo) |
| `e5.py prog` | `res_e5prog.json` | Table 7 (CEV rows) |
| `e6.py` | `res_e6.json` | put-call check (Section 3.3), absorption probabilities (Remark 2.3), Figure 3 |
| `e7.py wp` | `res_e7wp.json` | Figure 4 |
| `e7.py gbm`, `cev`, `curransmile`, `apocontract` | `res_e7gbm.json`, `res_e7cev.json` | Table 8 |
| `e7.py mctime` | `res_e7mct.json` | Monte Carlo cost in Section 5.4 |
| `e8.py calib` | `res_e8calib.json` | Figure 5(c) |
| `e8.py conv` | `res_e8conv.json` | convergence statements in Section 5.5 |
| `e8.py price` | `res_e8price.json` | Table 10 (PDE and closed-form approximations) |
| `e8.py mc4`, `mc4_201`, `mc4_202`, `mc4_203` | `res_e8mc4*.json` | Monte Carlo with step 1/1008 (bias check, Section 5.5) |
| `e8.py mc8`, `mc8_301`, `mc8_302`, `mc8_303` | `res_e8mc8*.json` | Table 10 (Monte Carlo reference, step 1/2016) |
| `greeks.py brent`, `greeks.py strip` | `res_e9brent.json`, `res_e9strip.json` | Table 11 |
| `timing.py` | `res_e10time.json` | Table 12 and the CPU times quoted in Sections 5.4-5.7 |
| `text_numbers.py` | `res_textnumbers.json` | every number quoted in the text that is not in a table or `macros.tex` (prints each with the sentence that quotes it) |

Steps that must run in order: `e1`, `e2` and `e4` before `e7.py gbm`/`wp`; `e7.py cev` before
`curransmile` and `apocontract`. `run_all.py` uses this order.

## Tables, figures and macros

| script | writes (in `paper/`) |
|---|---|
| `mk.py` | `tab_bs`, `tab_conv`, `tab_curve` (Table 3, from the curve in `common.py`), `tab_apo_bs`, `tab_cev`, `tab_apo_cev`, `tab_prog` (Tables 1-7, read by `\input`), `fig_conv.pdf` (Figure 1), `fig_skew.pdf` (Figure 3), `macros.tex` (numbers quoted in the text) |
| `mk2.py` | `tab_compare`, `tab_strip`, `tab_greeks`, `tab_cpu` (Tables 8, 10, 11, 12; these are pasted inline in the paper) |
| `fig_wp.py` | `fig_wp.pdf` (Figure 4) |
| `fig_market.py` | `fig_market.pdf` (Figure 5) |

Table 9 (the natural-gas curve) is typed in the paper; its values are the inputs in `realistic.py`.

## Folders

* `reference_results/`: the results used in the paper.
* `paper/`: the LaTeX source; generated tables, figures and `macros.tex` are written here.
* `diagnostics/`: an independent re-implementation of Algorithm 1 for GBM (`apo_k110.py`) and the
  scripts used to investigate the larger error at K=110 (`conv.py`, `scanK.py`). They are not needed for
  the paper; Figure 2 is produced with the main solver and agrees with this re-implementation to
  4e-12.
