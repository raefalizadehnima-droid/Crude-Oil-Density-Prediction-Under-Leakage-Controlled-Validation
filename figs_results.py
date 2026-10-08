import numpy as np, pandas as pd, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from lib import get_data
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
d=get_data();y=d['y'];Z=np.load('out/oof_main_part0.npz');T=pd.read_csv('out/table_performance.csv');C=pd.read_csv('out/table_comparisons.csv')
e1=json.load(open('out/extra1.json'));e2=json.load(open('out/extra2.json'));pp=json.load(open('out/pp_comparison.json'))
# preprocessing
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
for a,b in zip(ax,('NIR','FTIR','MS')):
    ks=[k for k in pp if k.startswith(b+'|') and k.endswith('PLS')]; kr=[k.replace('PLS','Ridge') for k in ks]
    lab=['|'.join(k.split('|')[1:3]) for k in ks];x=np.arange(len(ks))
    a.bar(x-.2,[pp[k]['R2'] for k in ks],.4,yerr=[pp[k]['R2sd'] for k in ks],label='PLS',color='#4c72b0');a.bar(x+.2,[pp[k]['R2'] for k in kr],.4,yerr=[pp[k]['R2sd'] for k in kr],label='Ridge',color='#dd8452')
    a.set_xticks(x);a.set_xticklabels(lab,rotation=60,ha='right',fontsize=7);a.set_ylabel('Nested-CV R$^2$');a.set_title(b)
ax[0].legend(frameon=False);plt.tight_layout();plt.savefig('out/figs/fig_preproc.png',dpi=200);plt.close()
# heatmap
order=['NIR','FTIR','MS','MSdesc','NIR+FTIR','MS+NIR','MS+FTIR','ALL'];models=['PLS','Ridge','ENet','SVR','ET','XGB','Late-mean','Late-stack','SO-PLS']
H=np.full((len(order),len(models)),np.nan)
for _,r in T.iterrows():
    if r.featureset in order and r.model in models: H[order.index(r.featureset),models.index(r.model)]=r.R2
fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(H,cmap='viridis',vmin=0,vmax=1)
ax.set_xticks(range(len(models)));ax.set_xticklabels(models,rotation=45,ha='right');ax.set_yticks(range(len(order)));ax.set_yticklabels(order)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if not np.isnan(H[i,j]): ax.text(j,i,f'{H[i,j]:.2f}',ha='center',va='center',color='w' if H[i,j]<.6 else 'k',fontsize=7)
plt.colorbar(im,label='Nested-CV R$^2$ (mean of 5 repeats)');plt.tight_layout();plt.savefig('out/figs/fig_heatmap.png',dpi=200);plt.close()
# pred vs obs
keys=[('FTIR|SVR','FTIR, SVR'),('ALL|SO-PLS','ALL, SO-PLS'),('MS|ENet','MS, ENet'),('NIR|Ridge','NIR, Ridge')]
fig,ax=plt.subplots(1,4,figsize=(12,3.1))
for a,(k,t) in zip(ax,keys):
    P=Z[k].mean(0);m=T[(T.featureset==k.split('|')[0])&(T.model==k.split('|')[1])].iloc[0]
    a.scatter(y,P,s=18,c='#4c72b0');lo,hi=y.min()-.01,y.max()+.01;a.plot([lo,hi],[lo,hi],'k--',lw=.8)
    a.set_title(f'{t}\nR$^2$={m.R2:.3f}, RMSE={m.RMSE:.4f}',fontsize=8);a.set_xlabel('Observed density');a.set_ylabel('Predicted (mean of repeats)')
    if k=='FTIR|SVR':
        for i in (33,28): a.annotate(str(i+1),(y[i],P[i]),fontsize=7,xytext=(3,3),textcoords='offset points')
plt.tight_layout();plt.savefig('out/figs/fig_predobs.png',dpi=200);plt.close()
# residuals
P=Z['FTIR|SVR'].mean(0);e=P-y;fig,ax=plt.subplots(1,4,figsize=(12,3))
ax[0].scatter(P,e,s=18);ax[0].axhline(0,c='k',lw=.8);ax[0].set_xlabel('Predicted');ax[0].set_ylabel('Residual (pred - obs)')
ax[1].hist(e,bins=12,color='#4c72b0',edgecolor='w');ax[1].set_xlabel('Residual')
stats.probplot(e,plot=ax[2]);ax[2].set_title('')
ax[3].bar(np.arange(1,43),e,color='#4c72b0');ax[3].set_xlabel('Sample (row index)');ax[3].set_ylabel('Residual')
plt.tight_layout();plt.savefig('out/figs/fig_residuals.png',dpi=200);plt.close()
# fusion comparison with CIs
fig,ax=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
fs=['NIR','FTIR','MS','NIR+FTIR','MS+NIR','MS+FTIR','ALL']
for a,mn in zip(ax,('PLS','Ridge','SO-PLS / Late-stack')):
    for i,f in enumerate(fs):
        if mn.startswith('SO'):
            if f in('NIR','FTIR','MS'): row=T[(T.featureset==f)&(T.model=='PLS')]
            else: row=T[(T.featureset==f)&(T.model=='SO-PLS')]
        else: row=T[(T.featureset==f)&(T.model==mn)]
        r=row.iloc[0];a.errorbar(i,r.RMSE,yerr=[[r.RMSE-r.RMSE_lo],[r.RMSE_hi-r.RMSE]],fmt='o',color='#4c72b0',capsize=3)
    a.set_xticks(range(len(fs)));a.set_xticklabels(fs,rotation=45,ha='right');a.set_title(mn+(' (single blocks: PLS)' if mn.startswith('SO') else ''),fontsize=8)
ax[0].set_ylabel('Nested-CV RMSE (95% bootstrap CI over samples)');plt.tight_layout();plt.savefig('out/figs/fig_fusion.png',dpi=200);plt.close()
# forest
fig,ax=plt.subplots(figsize=(7,5));lab=[f'{a} vs {b}' for a,b in zip(C.A,C.B)]
ax.errorbar(C.dRMSE,range(len(C)),xerr=[C.dRMSE-C.ci_lo,C.ci_hi-C.dRMSE],fmt='o',capsize=3,color='#4c72b0');ax.axvline(0,c='k',lw=.8)
ax.set_yticks(range(len(C)));ax.set_yticklabels(lab,fontsize=7);ax.invert_yaxis();ax.set_xlabel('$\\Delta$RMSE (A - B), 95% bootstrap CI');plt.tight_layout();plt.savefig('out/figs/fig_forest.png',dpi=200);plt.close()
# robustness
fig,ax=plt.subplots(1,4,figsize=(14,3.2))
for nm,c in (('FTIR','#4c72b0'),('ALL','#dd8452')):
    v=e1['perm_'+nm];ax[0].hist(v['null_R2'],bins=10,alpha=.6,color=c,label=nm+' null');ax[0].axvline(v['obs_R2_rep0'],color=c,lw=2)
ax[0].set_xlabel('Nested-CV R$^2$ (permuted y; line = observed)');ax[0].legend(frameon=False,fontsize=7)
lc=e1['learning_curve']
for (nm,mn),c in zip((('FTIR','Ridge'),('MS','Ridge'),('ALL','Ridge'),('FTIR','PLS')),('#4c72b0','#55a868','#dd8452','#8172b2')):
    fr=[.4,.6,.8,1.0];ax[1].plot([f*36 for f in fr],[lc[f'{nm}|{mn}|{f}']['RMSE'][0] for f in fr],'o-',label=f'{nm} {mn}',color=c)
ax[1].set_xlabel('Training samples');ax[1].set_ylabel('RMSE');ax[1].legend(frameon=False,fontsize=7)
bw=e1['block_weight']
for blk,c in (('MS','#55a868'),('NIR','#c44e52')):
    w=[.25,.5,1,2,4];ax[2].plot(w,[bw[f'{blk}|{x}']['RMSE'][0] for x in w],'o-',label=blk+' weight',color=c)
ax[2].set_xscale('log',base=2);ax[2].set_xlabel('Block weight (FTIR = 1)');ax[2].set_ylabel('RMSE (ALL early PLS)');ax[2].legend(frameon=False,fontsize=7)
nz=e2['noise'];lv=[0,.01,.03,.1,.3]
for nm,c in (('FTIR','#4c72b0'),('MS','#55a868'),('ALL','#dd8452')): ax[3].plot(lv,[nz[f'{nm}|{x}']['RMSE'][0] for x in lv],'o-',label=nm,color=c)
ax[3].set_xlabel('Test-time noise (fraction of median feature SD)');ax[3].set_ylabel('RMSE (PLS)');ax[3].legend(frameon=False,fontsize=7)
plt.tight_layout();plt.savefig('out/figs/fig_robust.png',dpi=200);plt.close();print('ok')
