import numpy as np
from scipy.optimize import nnls
from lib import *

# ---------- engineered MS descriptors (derived only from m/z and intensity; no formula assignment) ----------
def ms_descriptors(M,mz):
    I=M; rel=I/I.sum(1,keepdims=True)
    f={}
    edges=np.arange(140,1040,20)
    for lo in edges[:-1]:
        s=(mz>=lo)&(mz<lo+20); f[f'rel_{lo}-{lo+20}']=rel[:,s].sum(1)*100
    w=rel; mw=(w*mz).sum(1)                       # intensity-weighted mean m/z
    mn=1.0/((w/mz).sum(1))                        # number-average-like
    sd=np.sqrt((w*(mz-mw[:,None])**2).sum(1))
    sk=(w*(mz-mw[:,None])**3).sum(1)/sd**3
    f['mz_wmean']=mw;f['mz_nmean']=mn;f['mz_PDI']=mw/mn;f['mz_wsd']=sd;f['mz_wskew']=sk
    f['frac_mz_gt500']=rel[:,mz>500].sum(1);f['frac_mz_lt250']=rel[:,mz<250].sum(1)
    km=mz*14.0/14.01565; kmd=np.round(km)-km        # Kendrick mass defect (CH2 base) - a transform of m/z only
    for lo in np.arange(-0.30,0.30,0.05):
        s=(kmd>=lo)&(kmd<lo+0.05); f[f'kmd_{lo:.2f}']=rel[:,s].sum(1)*100
    f['kmd_wmean']=(w*kmd).sum(1)
    f['frac_mass_defect_wmean']=(w*(mz-np.floor(mz))).sum(1)
    names=list(f); return np.column_stack([f[n] for n in names]),names

# ---------- SO-PLS (sequential, orthogonalised) ----------
def _chain(Xtr,ytr,Xte,comps,sweep_last=None):
    mu_y=ytr.mean(); res=ytr-mu_y; pred=np.full(Xte[0].shape[0],mu_y); Ttr=None;Tte=None
    for j in range(len(Xtr)):
        mu=Xtr[j].mean(0); Xa=Xtr[j]-mu; Xb=Xte[j]-mu
        if Ttr is not None:
            G=np.linalg.lstsq(Ttr,Xa,rcond=None)[0]; Xa=Xa-Ttr@G; Xb=Xb-Tte@G
        if sweep_last==j:
            m=pls1_fit(Xa,res-res.mean(),comps[j]); return pred[None,:]+(Xb@m['B'].T).T   # (A,nte)
        k=comps[j]
        if k==0: continue
        m=pls1_fit(Xa,res-res.mean(),k); B=m['B'][k-1]
        res=res-Xa@B; pred=pred+Xb@B
        Rm=m['W'][:,:k]@np.linalg.inv(m['P'][:,:k].T@m['W'][:,:k])
        ta,tb=Xa@Rm,Xb@Rm
        Ttr=ta if Ttr is None else np.hstack([Ttr,ta]); Tte=tb if Tte is None else np.hstack([Tte,tb])
    return pred
def fit_predict_soplS(Xtr,ytr,Xte,seed,Amax=8):
    nb=len(Xtr); n=len(ytr); comps=[0]*nb
    folds=inner_kfold(n,seed)
    for j in range(nb):
        A=min(Amax,n-4); errs=np.zeros(A+1)
        for tri,vai in folds:
            xt=[x[tri] for x in Xtr]; xv=[x[vai] for x in Xtr]
            c=comps.copy(); c[j]=A
            P=_chain(xt,ytr[tri],xv,c,sweep_last=j)               # (A,nv) predictions for k=1..A
            P0=_chain(xt,ytr[tri],xv,[comps[i] if i<j else 0 for i in range(nb)])  # k=0 for block j
            allp=np.vstack([P0[None,:] if P0.ndim==1 else P0,P])
            errs+=((allp-ytr[vai][None,:])**2).sum(1)
        comps[j]=int(np.argmin(errs))
    return _chain(Xtr,ytr,Xte,comps),dict(comps=comps)

# ---------- late fusion of unimodal PLS ----------
def fit_predict_late(Xtr,ytr,Xte,seed,Amax=12):
    n=len(ytr); folds=inner_kfold(n,seed); preds=[];oofs=[]
    for Xa,Xb in zip(Xtr,Xte):
        p,info=fit_predict_pls(Xa,ytr,Xb,seed,Amax); k=info['k']; preds.append(p)
        oof=np.zeros(n)
        for tri,vai in folds:
            mu=Xa[tri].mean(0); m=pls1_fit(Xa[tri]-mu,ytr[tri]-ytr[tri].mean(),Amax)
            oof[vai]=(Xa[vai]-mu)@m['B'][min(k,m['B'].shape[0])-1]+ytr[tri].mean()
        oofs.append(oof)
    preds=np.array(preds);oofs=np.array(oofs); ym=ytr.mean()
    mean_pred=preds.mean(0)
    w,_=nnls((oofs-ym).T,ytr-ym); stack=ym+(w[:,None]*(preds-ym)).sum(0)
    return mean_pred,stack,dict(w=w.tolist())
