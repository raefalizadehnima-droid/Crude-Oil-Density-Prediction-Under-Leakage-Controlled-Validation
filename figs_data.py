import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, json
from scipy import stats
from sklearn.decomposition import PCA
from extra_common import *
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
cm=plt.cm.viridis; norm=plt.Normalize(y.min(),y.max())
# Fig 3: target EDA
fig,ax=plt.subplots(1,3,figsize=(10,3))
ax[0].hist(y,bins=12,color='#4c72b0',edgecolor='w');ax[0].set_xlabel('Density (units as supplied)');ax[0].set_ylabel('Number of oils')
stats.probplot(y,plot=ax[1]);ax[1].set_title('')
ax[2].plot(np.arange(1,43),y,'o',color='#4c72b0');ax[2].set_xlabel('Row index (sample)');ax[2].set_ylabel('Density')
for a,t in zip(ax,'abc'): a.text(-0.12,1.05,t,transform=a.transAxes,fontweight='bold')
plt.tight_layout();plt.savefig('out/figs/fig_target.png',dpi=200);plt.close()
# Fig 4: MS overview
M=d['MS'];fig,ax=plt.subplots(1,3,figsize=(11,3.2))
mean=M.mean(0);ax[0].vlines(mz,0,mean/1e9*0+np.sqrt(mean)/1e3,lw=.2,color='#4c72b0');ax[0].set_xlabel('m/z');ax[0].set_ylabel('sqrt(mean intensity) / 10$^3$')
ax[1].hist(np.log10(M.ravel()),bins=60,color='#55a868');ax[1].set_xlabel('log$_{10}$ intensity');ax[1].set_ylabel('Peak-sample entries')
rel=M/M.sum(1,keepdims=True);mw=(rel*mz).sum(1)
ax[2].scatter(mw,y,c=y,cmap=cm,s=25);r=np.corrcoef(mw,y)[0,1];ax[2].set_xlabel('Intensity-weighted mean m/z');ax[2].set_ylabel('Density');ax[2].set_title(f'Pearson r = {r:.2f} (descriptive, n = 42)',fontsize=8)
for a,t in zip(ax,'abc'): a.text(-0.12,1.05,t,transform=a.transAxes,fontweight='bold')
plt.tight_layout();plt.savefig('out/figs/fig_ms.png',dpi=200);plt.close()
json.dump(dict(r_mwmz_density=float(r),mw_min=float(mw.min()),mw_max=float(mw.max())),open('out/ms_desc.json','w'))
# Fig 5: spectra
fig,ax=plt.subplots(2,2,figsize=(10,6))
for i,y_i in enumerate(y):
    c=cm(norm(y_i))
    ax[0,0].plot(d['wn_nir'],d['NIR'][i],color=c,lw=.6);ax[0,1].plot(d['wn_ftir'],d['FTIR'][i],color=c,lw=.6)
    ax[1,0].plot(d['wn_nir'],RAW['NIR'][i],color=c,lw=.6);ax[1,1].plot(d['wn_ftir'],RAW['FTIR'][i],color=c,lw=.6)
ax[0,0].set_title('NIR, raw');ax[0,1].set_title('FTIR, raw');ax[1,0].set_title('NIR, MSC');ax[1,1].set_title('FTIR, SNV + 1st derivative (SG)')
for a in ax.ravel(): a.set_xlabel('Wavenumber / cm$^{-1}$');a.invert_xaxis()
ax[0,0].set_ylabel('Absorbance (as supplied)');ax[0,1].set_ylabel('Absorbance (as supplied)')
fig.subplots_adjust(hspace=0.5,wspace=0.3)
sm=plt.cm.ScalarMappable(norm=norm,cmap=cm);fig.colorbar(sm,ax=ax,label='Density',shrink=.6)
plt.savefig('out/figs/fig_spectra.png',dpi=200,bbox_inches='tight');plt.close()
# Fig 6: PCA of each block (preprocessed, centred) + scree
fig,ax=plt.subplots(2,3,figsize=(11,6)); info={}
for j,(b,X) in enumerate((('NIR',RAW['NIR']),('FTIR',RAW['FTIR']),('MS',RAW['MS']))):
    Xc=X-X.mean(0); p=PCA(10).fit(Xc);S=p.transform(Xc);ev=p.explained_variance_ratio_
    ax[0,j].scatter(S[:,0],S[:,1],c=y,cmap=cm,s=28);ax[0,j].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)');ax[0,j].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)');ax[0,j].set_title(b+' (as used for modelling)')
    ax[1,j].bar(range(1,11),ev*100,color='#4c72b0');ax[1,j].set_xlabel('Component');ax[1,j].set_ylabel('Explained variance / %')
    r=[stats.spearmanr(S[:,k],y)[0] for k in range(5)]
    info[b]=dict(ev=ev[:5].tolist(),spearman_pc_density=r)
json.dump(info,open('out/pca.json','w'),default=float)
plt.tight_layout();plt.savefig('out/figs/fig_pca.png',dpi=200);plt.close()
print(json.dumps(info,default=float,indent=0)); print(r)
