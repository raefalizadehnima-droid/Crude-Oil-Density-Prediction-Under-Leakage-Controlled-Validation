import numpy as np,pandas as pd,json
from common import *
from scipy import stats
y,N,F,M=load()
R={}
# target
R['y']=dict(n=len(y),mean=y.mean(),sd=y.std(),min=y.min(),max=y.max(),median=y.median(),skew=stats.skew(y),kurt=stats.kurtosis(y),shapiro_p=stats.shapiro(y).pvalue,n_unique=y.nunique())
q1,q3=y.quantile([.25,.75]);iqr=q3-q1
R['y_iqr_outliers']=y[(y<q1-1.5*iqr)|(y>q3+1.5*iqr)].to_dict()
# API gravity from density (standard definition, 15.56C SG)
api=141.5/y-131.5; R['api_min']=api.min();R['api_max']=api.max()
# replicate / near-duplicate detection
for n,d in [('NIR',N),('FTIR',F),('MS',M)]:
    X=d.values if n!='MS' else np.log10(d.values)
    C=np.corrcoef(X);np.fill_diagonal(C,np.nan)
    i,j=np.unravel_index(np.nanargmax(C),C.shape)
    Z=(X-X.mean(0))/(X.std(0)+1e-12)
    C2=np.corrcoef(Z);np.fill_diagonal(C2,np.nan)
    R[n+'_max_pair_corr']=float(np.nanmax(C));R[n+'_pair']=(int(i+1),int(j+1))
    R[n+'_max_pair_corr_standardized']=float(np.nanmax(C2))
    # outlier by Mahalanobis in PCA 3 comps
    from sklearn.decomposition import PCA
    S=PCA(3).fit_transform(Z); 
    md=np.sqrt(((S/S.std(0))**2).sum(1)); R[n+'_top_md']=[(int(k+1),round(float(md[k]),2)) for k in np.argsort(-md)[:4]]
# MS structure
cols=M.columns.astype(str); base=np.array([float(c.split('_')[0]) for c in cols])
vc=pd.Series(base).value_counts()
R['MS']=dict(n_cols=len(cols),n_unique_mz=int(vc.size),max_dup=int(vc.max()),frac_cols_with_dup_mz=float((pd.Series(base).map(vc)>1).mean()),mz_min=base.min(),mz_max=base.max(),
 int_min=float(M.values.min()),int_max=float(M.values.max()),int_median=float(np.median(M.values)),
 log10_skew=float(stats.skew(np.log10(M.values).ravel())),raw_skew=float(stats.skew(M.values.ravel())),
 rowsum_cv=float(M.sum(axis=1).std()/M.sum(axis=1).mean()))
# per-column: CV across samples
cv=(M.std()/M.mean());R['MS']['cv_median']=float(cv.median());R['MS']['cv_min']=float(cv.min());R['MS']['cv_max']=float(cv.max())
# feature-target corr summary (full-data, descriptive only)
for n,d in [('NIR',N),('FTIR',F),('MS',np.log10(M))]:
    r=d.apply(lambda c: np.corrcoef(c,y)[0,1]); R[n+'_maxabs_r_y']=float(r.abs().max()); R[n+'_argmax']=str(r.abs().idxmax())
# mz spacing
u=np.sort(vc.index.values);R['MS']['median_gap_mz']=float(np.median(np.diff(u)))
# kendrick mass defect cannot be computed w/o assignment, but nominal mass defect is derivable
nm=np.round(base); R['MS']['frac_mass_defect_mean']=float((base-np.floor(base)).mean())
# mean spectra stats
R['NIR']=dict(p=N.shape[1],range=(float(N.columns.astype(float).min()),float(N.columns.astype(float).max())),step=0.5,absmin=float(N.values.min()),absmax=float(N.values.max()))
R['FTIR']=dict(p=F.shape[1],range=(float(F.columns.astype(float).min()),float(F.columns.astype(float).max())),step=1.0,absmin=float(F.values.min()),absmax=float(F.values.max()))
# collinearity: adjacent-variable corr, effective dimension (PCA 99%)
from sklearn.decomposition import PCA
for n,d in [('NIR',N.values),('FTIR',F.values),('MS',np.log10(M.values))]:
    Z=d-d.mean(0); ev=PCA().fit(Z).explained_variance_ratio_
    R[n+'_pc_for_95']=int(np.searchsorted(np.cumsum(ev),.95)+1);R[n+'_pc_for_99']=int(np.searchsorted(np.cumsum(ev),.99)+1);R[n+'_pc1']=float(ev[0])
# n/p
R['features_total']=N.shape[1]+F.shape[1]+M.shape[1]; R['p_over_n']=R['features_total']/42
print(json.dumps(R,indent=1,default=str))
json.dump(R,open('out/eda.json','w'),indent=1,default=str)
