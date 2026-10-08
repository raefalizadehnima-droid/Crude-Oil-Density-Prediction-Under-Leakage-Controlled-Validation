import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from extra_common import *
from lib import _make_xgb
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
e=json.load(open('out/extra3.json'));I=np.load('out/interp_pls.npz');off=I['off'];pa=I['phi_abs'];beta=I['beta']
# Fig: block importance + stability + region contributions
fig=plt.figure(figsize=(12,6.2));gs=fig.add_gridspec(2,3)
a=fig.add_subplot(gs[0,0]);bp=e['block_perm_importance']['imp'];bl=['FTIR','MS','NIR']
a.bar(bl,[bp[b]['mean'] for b in bl],yerr=[bp[b]['sd'] for b in bl],color=['#4c72b0','#55a868','#c44e52']);a.axhline(0,c='k',lw=.8);a.set_ylabel('$\\Delta$RMSE on block permutation');a.set_title('Block permutation importance\n(ALL early-fusion PLS)',fontsize=8)
a=fig.add_subplot(gs[0,1]);st=e['coef_stability'];a.bar(bl,[st[b]['mean'] for b in bl],color=['#4c72b0','#55a868','#c44e52']);a.set_ylabel('Mean pairwise r of PLS coefficients\nacross outer folds');a.set_ylim(0,1);a.set_title('Coefficient stability',fontsize=8)
a=fig.add_subplot(gs[0,2]);ax=I['wn_nir'];s=pa[off[2]:];a.plot(ax,s/s.sum()*len(s),color='#c44e52',lw=.6);a.invert_xaxis();a.set_xlabel('NIR wavenumber / cm$^{-1}$');a.set_ylabel('Relative mean |contribution|');a.set_title('NIR (final PLS; in-sample)',fontsize=8)
a=fig.add_subplot(gs[1,:2]);ax=I['wn_ftir'];s=pa[off[0]:off[1]];a.plot(ax,s/s.max(),color='#4c72b0',lw=.8)
for lo,hi,t in ((2800,3000,'C–H stretch'),(1400,1550,'CH$_2$/CH$_3$ bend'),(800,900,'aromatic C–H oop')): a.axvspan(lo,hi,color='#dd8452',alpha=.25);a.text((lo+hi)/2,1.02,t,ha='center',fontsize=7)
a.invert_xaxis();a.set_xlabel('FTIR wavenumber / cm$^{-1}$');a.set_ylabel('Mean |contribution| (rel.)');a.set_title('FTIR (final early-fusion PLS; in-sample)',fontsize=8,pad=14)
a=fig.add_subplot(gs[1,2]);mz=I['mz'];s=pa[off[1]:off[2]];a.vlines(mz,0,s/s.max(),lw=.3,color='#55a868');a.set_xlabel('m/z');a.set_ylabel('Mean |contribution| (rel.)');a.set_title('MS peaks (final PLS; in-sample)',fontsize=8)
plt.tight_layout();plt.savefig('out/figs/fig_interp_pls.png',dpi=200);plt.close()
# SHAP summary + dependence + PDP
X=np.load('out/interp_xgb.npz',allow_pickle=True);sv=X['sv'];F=X['X'];nm=[str(n) for n in X['names']];imp=np.abs(sv).mean(0);o=np.argsort(-imp)[:12]
rng=np.random.default_rng(0);fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.2,1]})
for r,j in enumerate(o[::-1]):
    v=F[:,j];c=(v-v.min())/(np.ptp(v)+1e-12);ax[0].scatter(sv[:,j],r+rng.uniform(-.25,.25,len(v)),c=c,cmap='coolwarm',s=14)
ax[0].set_yticks(range(len(o)));ax[0].set_yticklabels([nm[j].replace('@',' @ ')+(' cm$^{-1}$' if nm[j].startswith(('FTIR','NIR')) else '') for j in o[::-1]],fontsize=7);ax[0].axvline(0,c='k',lw=.6);ax[0].set_xlabel('SHAP value (density units)')
ax[0].set_title('SHAP summary: XGBoost on pooled features\n(fit to all 42 oils; interpretation only)',fontsize=8)
y_=y;mdl=_make_xgb(0).fit(F,y_)
for a,j in zip(ax[1:],o[:1]):
    pass
j=o[0];grid=np.linspace(F[:,j].min(),F[:,j].max(),30);pd_=[]
for g in grid:
    Xg=F.copy();Xg[:,j]=g;pd_.append(mdl.predict(Xg).mean())
ax[1].plot(grid,pd_,color='#4c72b0',label=nm[j]);ax[1].scatter(F[:,j],sv[:,j]+mdl.predict(F).mean(),s=10,color='gray',alpha=.6,label='SHAP + mean prediction')
j2=o[1];grid2=np.linspace(F[:,j2].min(),F[:,j2].max(),30);pd2=[]
for g in grid2:
    Xg=F.copy();Xg[:,j2]=g;pd2.append(mdl.predict(Xg).mean())
ax[1].set_xlabel('Pooled preprocessed feature value (centred)');ax[1].set_ylabel('Partial dependence of density');ax[1].legend(frameon=False,fontsize=7);ax[1].set_title(f'Partial dependence: {nm[j]}',fontsize=8)
plt.tight_layout();plt.savefig('out/figs/fig_shap.png',dpi=200);plt.close()
# workflow schematic
fig,ax=plt.subplots(figsize=(12,4.3));ax.axis('off')
def box(x,y0,w,h,t,c='#dbe6f4'):
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=0.02',fc=c,ec='#444'));ax.text(x+w/2,y0+h/2,t,ha='center',va='center',fontsize=7.5)
box(0,2.6,2.2,.9,'NIR\n11,101 pts\n(n = 42)');box(0,1.5,2.2,.9,'FTIR\n3,201 pts\n(n = 42)');box(0,.4,2.2,.9,'APPI(+) FT-ICR MS\n23,800 peaks\n(n = 42)');box(0,3.7,2.2,.6,'Density (target)','#f6dcc7')
box(2.8,2.6,2.4,.9,'Row-wise preprocessing\n(MSC in-fold)');box(2.8,1.5,2.4,.9,'SNV + SG 1st deriv.\n(stateless)');box(2.8,.4,2.4,.9,'sqrt intensity\n(stateless)')
box(5.8,1.0,2.6,2.6,'Fusion strategies\n\nEarly: block-scaled concat\nLate: mean / stacked PLS\nSequential: SO-PLS\n(+ unimodal controls)','#e2f0d9')
box(9.0,2.3,2.8,1.3,'Learners: PLS, Ridge, ENet,\nSVR, ExtraTrees, XGBoost\n(inner 5-fold tuning)');box(9.0,.5,2.8,1.3,'Outer CV: 5 x 6-fold,\ntarget-stratified;\nall fitting in-fold','#fde9c9')
for yy in (3.05,1.95,.85): ax.annotate('',(2.8,yy),(2.2,yy),arrowprops=dict(arrowstyle='->'));ax.annotate('',(5.8,yy+ (0 if yy==1.95 else (-.2 if yy>2 else .3))),(5.2,yy),arrowprops=dict(arrowstyle='->'))
ax.annotate('',(9.0,2.9),(8.4,2.4),arrowprops=dict(arrowstyle='->'));ax.annotate('',(10.4,1.8),(10.4,2.3),arrowprops=dict(arrowstyle='->'))
ax.text(5.9,.3,'Explainability: block permutation, PLS coefficients/VIP, SHAP, PDP\nRobustness: label permutation, learning curve, block weight, noise, LOOCV',fontsize=7.5,va='top')
ax.set_xlim(-.2,12);ax.set_ylim(-.4,4.5);plt.savefig('out/figs/fig_workflow.png',dpi=200,bbox_inches='tight');plt.close();print('ok')
