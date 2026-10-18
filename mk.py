import json, numpy as np, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import fut_days, fut_px, curve_b
L=lambda n: json.load(open(f'res_{n}.json'))
P='paper/'; mac={}
f=lambda x,d=4: f'{x:.{d}f}'
def e(x): return f'{x:+.1e}'.replace('e-0','e-').replace('e+0','e+')
def sci(x):
    m,ex=f'{x:.1e}'.split('e'); return f'${m}\\times10^{{{int(ex)}}}$'
def dif(x): 
    m,ex=f'{abs(x):.1e}'.split('e'); s='-' if x<0 else ''
    return f'${s}{m}\\mathrm{{e}}{{{int(ex)}}}$'
# ---- Table BS
E1=L('e1'); rows=[]
for d in E1:
    rows.append(f"{d['sig']:.1f} & {d['K']} & {f(d['pde'],5)} & {dif(d['pde']-d['mc'])} & {f(d['vorst'],5)} & {dif(d['vorst']-d['mc'])} & {f(d['mc'],5)} & {d['se']:.1e} \\\\")
open(P+'tab_bs.tex','w').write("\\begin{tabular}{cc rr rr rr}\\toprule\n & & \\multicolumn{2}{c}{PDE} & \\multicolumn{2}{c}{Vorst} & \\multicolumn{2}{c}{Monte Carlo}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n$\\sigma$ & $K$ & Value & Error & Value & Error & Value & SE\\\\\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\\end{tabular}\n")
mac['cpubs']=f"{np.mean([d['cpu'] for d in E1]):.1f}"
mac['bsmaxrel']=f"{max(abs(d['pde']-d['mc'])/d['mc'] for d in E1):.0e}".replace('e-0','\\times10^{-')+'}'
# ---- Table conv
E2=L('e2'); Ks=['90','100','110']
ref={K:(4*E2[K][-1][1]-E2[K][-2][1])/3 for K in Ks}
lines=[]; errs={K:[] for K in Ks}
for i in range(len(E2['90'])):
    Ny=E2['90'][i][0]; cells=[]
    for K in Ks:
        v=E2[K][i][1]; er=abs(v-ref[K]); errs[K].append(er)
        rate='' if i==0 else f"{math.log2(errs[K][i-1]/er):.2f}"
        last = i==len(E2['90'])-1
        cells.append(f"{v:.7f} & {'--' if last else f'{er:.1e}'} & {'--' if last else rate}")
    cpu=np.mean([E2[K][i][2] for K in Ks])
    lines.append(f"{Ny} & {math.ceil(Ny/12)} & "+" & ".join(cells)+f" & {cpu:.2f}\\\\")
lines.append('\\midrule')
for i in range(1,len(E2['90'])-1):
    cells=[]
    for K in Ks:
        v=(4*E2[K][i][1]-E2[K][i-1][1])/3; cells.append(f"{v:.7f} & {abs(v-ref[K]):.1e} & ")
    lines.append(f"R({E2['90'][i-1][0]},{E2['90'][i][0]}) & & "+" & ".join(cells)+" & \\\\")
lines.append('\\midrule Reference & & '+" & ".join(f"{ref[K]:.7f} & & " for K in Ks)+" & \\\\")
open(P+'tab_conv.tex','w').write("\\begin{tabular}{rr rrr rrr rrr r}\\toprule\n& & \\multicolumn{3}{c}{$K=90$} & \\multicolumn{3}{c}{$K=100$} & \\multicolumn{3}{c}{$K=110$} & \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n$N_y$ & $N_t$ & Value & Error & Rate & Value & Error & Rate & Value & Error & Rate & CPU\\\\\\midrule\n"+"\n".join(lines)+"\n\\bottomrule\\end{tabular}\n")
plt.figure(figsize=(5,3.6))
Ny=[r[0] for r in E2['90']]
for K,mk,fc,ms in zip(Ks,'osd',['none','C1','C2'],[9,5,6]):
    plt.loglog(Ny[:-1],errs[K][:-1],mk+'-',mfc=fc,ms=ms,label=f'K={K}')
plt.xticks(Ny[:-1],[str(n) for n in Ny[:-1]]); plt.minorticks_off()
x=np.array([250,4000]); plt.loglog(x,3e-1*(x/1.0)**-2*1e0*errs['100'][0]/(3e-1*250**-2),'k--',lw=1,label='slope $-2$')
plt.xlabel('$N_y$'); plt.ylabel('absolute error'); plt.legend(); plt.grid(alpha=.3,which='both'); plt.tight_layout()
plt.savefig(P+'fig_conv.pdf')
# ---- curve table
muk=curve_b.muk
open(P+'tab_curve.tex','w').write("\\begin{tabular}{l"+"r"*len(fut_days)+"}\\toprule\nExpiry (business day) & "+" & ".join(map(str,fut_days))+"\\\\\nFutures price $F_{T_k}(0)$ & "+" & ".join(f'{p:.2f}' for p in fut_px)+"\\\\\nKnot value $\\mu_k$ & "+" & ".join(f'{m:+.3f}' for m in muk[1:])+"\\\\\\bottomrule\\end{tabular}\n")
# ---- APO GBM
E4=L('e4'); rows=[]
for d in E4['gbm']:
    rows.append(f"{d['K']} & {f(d['pde'],5)} & {dif(d['pde']-d['mc'])} & {f(d['vorst'],5)} & {dif(d['vorst']-d['mc'])} & {f(d['mc'],5)} & {d['se']:.1e}\\\\")
open(P+'tab_apo_bs.tex','w').write("\\begin{tabular}{c rr rr rr}\\toprule\n& \\multicolumn{2}{c}{PDE} & \\multicolumn{2}{c}{Vorst} & \\multicolumn{2}{c}{Monte Carlo}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n$K$ & Value & Error & Value & Error & Value & SE\\\\\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\\end{tabular}\n")
mac['cpuapo']=f"{np.mean([d['cpu'] for d in E4['gbm']]):.1f}"
# ---- CEV Asian
E3=L('e3'); M5={(d['sa'],d['beta'],d['K']):d for d in L('e5_0.2')+L('e5_0.4')}
rows=[]; bias=[]; sp=[]; br=[]; sk=[]; wins=0; ratio=[]
for d in E3:
    m=M5[(d['sa'],d['beta'],d['K'])]; mc=m['mc']
    bias.append(abs(d['mc']-mc)); es=d['spot']-mc; eb=d['bridge']-mc; sp.append(abs(es)); br.append(abs(eb)); sk.append(abs(d['gbm']-mc))
    wins+= abs(eb)<abs(es)
    rows.append(f"{d['sa']:.1f} & {d['beta']:.2f} & {d['K']} & {f(d['gbm'],4)} & {f(d['spot'],4)} & {es:+.4f} & {f(d['bridge'],4)} & {eb:+.4f} & {f(mc,4)} & {m['se']:.4f}\\\\")
    if d['K']==115 and d['beta']==0.75: rows.append('\\addlinespace')
open(P+'tab_cev.tex','w').write("\\begin{tabular}{ccc r rr rr rr}\\toprule\n& & & GBM & \\multicolumn{2}{c}{PDE, spot proxy} & \\multicolumn{2}{c}{PDE, bridge proxy} & \\multicolumn{2}{c}{Monte Carlo}\\\\\n\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n$\\sigma_{\\rm ATM}$ & $\\beta$ & $K$ & Value & Value & Error & Value & Error & Value & SE\\\\\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\\end{tabular}\n")
mac['cpucev']=f"{np.mean([d['c1'] for d in E3]):.1f}"; mac['cpucevb']=f"{np.mean([d['c2'] for d in E3]):.1f}"
mac['mcbias']=f"{max(bias):.4f}"
i=int(np.argmax(sk)); mac['skewmax']=f"{max(sk):.2f}"
d=E3[i]; mac['skewrel']=f"{100*sk[i]/M5[(d['sa'],d['beta'],d['K'])]['mc']:.0f}"
mac['spotmax']=f"{max(sp):.3f}"; j=int(np.argmax(sp)); mac['spotfrac']=f"{100*sp[j]/sk[j]:.0f}"
mac['brmax']=f"{max(br):.3f}"; mac['brwins']=str(wins)
mac['rmsratio']=f"{np.sqrt(np.mean(np.square(sp)))/np.sqrt(np.mean(np.square(br))):.1f}"
mac['maxratio']=f"{max(sp)/max(br):.1f}"
mac['brrelmax']=f"{100*max(b/M5[(d['sa'],d['beta'],d['K'])]['mc'] for b,d in zip(br,E3)):.2f}"
mac['sprelmax']=f"{100*max(b/M5[(d['sa'],d['beta'],d['K'])]['mc'] for b,d in zip(sp,E3)):.2f}"
# ---- APO CEV
A5=L('e5apo'); rows=[]
for d,m in zip(E4['cev'],A5):
    rows.append(f"{d['K']} & {f(d['gbm'],4)} & {f(d['spot'],4)} & {d['spot']-m['mc']:+.4f} & {f(d['bridge'],4)} & {d['bridge']-m['mc']:+.4f} & {f(m['mc'],4)} & {m['se']:.4f}\\\\")
open(P+'tab_apo_cev.tex','w').write("\\begin{tabular}{c r rr rr rr}\\toprule\n& GBM & \\multicolumn{2}{c}{PDE, spot proxy} & \\multicolumn{2}{c}{PDE, bridge proxy} & \\multicolumn{2}{c}{Monte Carlo}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n$K$ & Value & Value & Error & Value & Error & Value & SE\\\\\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\\end{tabular}\n")
mac['apocevmaxse']=f"{max(max(abs(d['spot']-m['mc']),abs(d['bridge']-m['mc']))/m['se'] for d,m in zip(E4['cev'],A5)):.1f}"
# ---- in-progress
P5={d['Fb1']:d for d in L('e5prog')}; rows=[]
for d in E4['prog']:
    sc=f"({d['Fb1']:.1f}, {d['F2']:.1f})"
    if d['model']=='gbm' or d['how']=='parity':
        rows.append(f"{sc} & {d['K']} & {d['Kp']:.3f} & {d['model'].upper()} & {'parity' if d['how']=='parity' else 'PDE'} & {f(d['price'],5)} & {dif(d['price']-d['mc'])} & {f(d['mc'],5)} & {d['se']:.1e}\\\\")
    else:
        m=P5[d['Fb1']]
        for lab,v in [('PDE, spot',m['spot']),('PDE, bridge',m['bridge'])]:
            rows.append(f"{sc} & {d['K']} & {d['Kp']:.3f} & CEV & {lab} & {f(v,5)} & {dif(v-m['mc'])} & {f(m['mc'],5)} & {m['se']:.1e}\\\\")
    if d['model']=='cev' and d['K']==80 and d['Fb1']<83: rows.append('\\addlinespace')
open(P+'tab_prog.tex','w').write("\\begin{tabular}{lcrll rr rr}\\toprule\n$(\\bar F_1,F_{T_2}(t))$ & $K$ & $K'$ & Model & Method & Value & Error & MC & SE\\\\\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\\end{tabular}\n")
# ---- skew fig
E6=L('e6'); sk6=E6['skew']; Kk=np.array(sk6['K'])
plt.figure(figsize=(5,3.6))
for b,mk in zip(['0.25','0.5','0.75'],'osd'): plt.plot(Kk,sk6['diff'][b],mk+'-',label=f'$\\beta={b}$')
plt.axhline(0,color='k',lw=.6); plt.axvline(sk6['fwd'],color='gray',ls=':')
plt.xlabel('strike $K$'); plt.ylabel('CEV $-$ GBM price'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(P+'fig_skew.pdf')
mac['fwdavg']=f"{sk6['fwd']:.2f}"
ab=E6['absorb']; mac['absmax']=sci(max(a[2] for a in ab)); mac['absbrent']=sci(ab[-1][2])
mac['parityres']=sci(max(abs(x[4]) for x in E6['putcall']))   # GBM, CEV spot and CEV bridge
open(P+'macros.tex','w').write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in mac.items())+"\n")
print(mac)
