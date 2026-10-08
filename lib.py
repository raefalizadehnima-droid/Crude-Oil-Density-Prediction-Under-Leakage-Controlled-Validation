import numpy as np, pandas as pd, time
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold, KFold
from common import load

SEED=42
# ---------- data ----------
def get_data():
    y,N,F,M=load()
    mz=np.array([float(str(c).split('_')[0]) for c in M.columns])
    return dict(y=y.values.astype(float), NIR=N.values.astype(float), FTIR=F.values.astype(float),
                MS=M.values.astype(float), mz=mz, wn_nir=N.columns.astype(float).values, wn_ftir=F.columns.astype(float).values)

# ---------- row-wise (stateless => leakage-free) preprocessing ----------
def snv(X): return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
def sg(X,w,d=1,p=2): return savgol_filter(X,w,p,deriv=d,axis=1)
def prep_spectrum(X, kind, kind_w):
    if kind=='raw': return X
    if kind=='snv': return snv(X)
    if kind=='sg1': return sg(X,kind_w,1)
    if kind=='snv_sg1': return sg(snv(X),kind_w,1)
    if kind=='sg2': return sg(X,kind_w,2)
    if kind=='msc': return X   # MSC reference learned in-fold (see Block)
    raise ValueError(kind)
def prep_ms(X, kind):
    if kind=='raw': return X
    if kind=='sqrt': return np.sqrt(X)
    if kind=='log': return np.log10(X)
    if kind=='rel_log': return np.log10(X/X.sum(1,keepdims=True)*1e6)
    raise ValueError(kind)

# ---------- block with in-fold learned parameters ----------
class Block:
    """kind: 'msc' or None; scale: 'center' | 'auto'. All parameters learned on the training rows only."""
    def __init__(self,X,msc=False,scale='center'): self.X=X; self.msc=msc; self.scale=scale
    def fit(self,tr):
        Xt=self.X[tr]
        if self.msc: self.ref=Xt.mean(0); Xt=self._msc(Xt)
        self.mu=Xt.mean(0); Xc=Xt-self.mu
        self.sd=Xc.std(0)+1e-12 if self.scale=='auto' else 1.0
        Xc=Xc/self.sd
        self.fro=np.sqrt((Xc**2).sum()/len(tr))   # block weight: unit mean-square norm
        return self
    def _msc(self,X):
        r=self.ref; rc=r-r.mean(); out=np.empty_like(X)
        for i in range(len(X)):
            b=np.dot(X[i]-X[i].mean(),rc)/np.dot(rc,rc); a=X[i].mean()-b*r.mean()
            out[i]=(X[i]-a)/b
        return out
    def transform(self,idx,block_scale=False):
        Xt=self.X[idx]
        if self.msc: Xt=self._msc(Xt)
        Xt=(Xt-self.mu)/self.sd
        return Xt/self.fro if block_scale else Xt

# ---------- PLS1 (all component counts at once) ----------
def pls1_fit(X,y,A):
    """X centred (n,p), y centred (n,). Returns B (A,p) cumulative coefficients, W,P,T,q."""
    n,p=X.shape; A=min(A,n-1)
    Xk=X.copy(); yk=y.copy(); W=np.zeros((p,A)); P=np.zeros((p,A)); q=np.zeros(A); T=np.zeros((n,A)); a_eff=0
    for a in range(A):
        w=Xk.T@yk; nw=np.linalg.norm(w)
        if nw<1e-14: break
        w/=nw; t=Xk@w; tt=t@t
        if tt<1e-14: break
        pp=Xk.T@t/tt; qa=(yk@t)/tt
        Xk-=np.outer(t,pp); yk=yk-qa*t
        W[:,a]=w;P[:,a]=pp;q[a]=qa;T[:,a]=t;a_eff=a+1
    B=np.zeros((A,p))
    for k in range(1,a_eff+1):
        Wk=W[:,:k];Pk=P[:,:k]
        B[k-1]=Wk@np.linalg.solve(Pk.T@Wk,q[:k])
    for k in range(a_eff,A): B[k]=B[a_eff-1]
    return dict(B=B,W=W,P=P,q=q,T=T,A=A)

def pls_vip(m,k):
    W=m['W'][:,:k];T=m['T'][:,:k];q=m['q'][:k]
    ss=(q**2)*(T**2).sum(0)
    Wn=W/np.linalg.norm(W,axis=0)
    return np.sqrt(W.shape[0]*((Wn**2)@ss)/ss.sum())

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def inner_kfold(n,seed,k=5): return list(KFold(k,shuffle=True,random_state=seed).split(np.arange(n)))

# ---------- tuned estimators operating on a matrix pair (already scaled) ----------
def fit_predict_pls(Xtr,ytr,Xte,seed,Amax=12,return_model=False):
    ym=ytr.mean(); yc=ytr-ym; n=len(ytr)
    A=min(Amax,n-2); errs=np.zeros(A); 
    for tri,vai in inner_kfold(n,seed):
        mu=Xtr[tri].mean(0); m=pls1_fit(Xtr[tri]-mu,yc[tri]-yc[tri].mean(),A)
        Bm=m['B']
        if Bm.shape[0]<A: Bm=np.vstack([Bm,np.repeat(Bm[-1:],A-Bm.shape[0],0)])
        pred=(Xtr[vai]-mu)@Bm.T+yc[tri].mean()
        errs+=((pred-yc[vai][:,None])**2).sum(0)
    k=int(np.argmin(errs))+1
    mu=Xtr.mean(0); m=pls1_fit(Xtr-mu,yc,A)
    pred=(Xte-mu)@m['B'][k-1]+ym
    if return_model: return pred,dict(k=k,model=m,mu=mu,ym=ym)
    return pred,dict(k=k)

def fit_predict_ridge(Xtr,ytr,Xte,seed,return_model=False):
    ym=ytr.mean(); n=len(ytr)
    def path(Xa,ya,Xb,rel):
        mu=Xa.mean(0); U,s,Vt=np.linalg.svd(Xa-mu,full_matrices=False); a=ya-ya.mean()
        Uy=U.T@a; Z=(Xb-mu)@Vt.T
        preds=[]
        for r in rel:
            lam=r*s[0]**2; preds.append(Z@((s/(s**2+lam))*Uy)+ya.mean())
        return np.array(preds)  # (nrel, nb)
    rel=10.0**np.arange(-8,0.5,0.5)
    errs=np.zeros(len(rel))
    for tri,vai in inner_kfold(n,seed):
        pr=path(Xtr[tri],ytr[tri],Xtr[vai],rel); errs+=((pr-ytr[vai][None,:])**2).sum(1)
    j=int(np.argmin(errs)); 
    pred=path(Xtr,ytr,Xte,rel[j:j+1])[0]
    return pred,dict(rel_alpha=float(rel[j]))

from sklearn.linear_model import ElasticNetCV
def fit_predict_enet(Xtr,ytr,Xte,seed):
    sc=Xtr.std()  # global scale only (keeps relative feature scaling)
    m=ElasticNetCV(l1_ratio=0.5,n_alphas=12,cv=inner_kfold(len(ytr),seed),max_iter=2000,tol=1e-3,fit_intercept=True,selection='cyclic')
    m.fit(Xtr/sc,ytr); 
    return m.predict(Xte/sc),dict(alpha=float(m.alpha_),nnz=int((m.coef_!=0).sum()))

from sklearn.svm import SVR
def _d2(A,B): return (A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T
def fit_predict_svr(Xtr,ytr,Xte,seed):
    n=len(ytr); ym,ys=ytr.mean(),ytr.std()
    grid=[(C,g) for C in (1,10,100) for g in (0.25,1,4)]
    def run(Xa,ya,Xb,params):
        mu=Xa.mean(0); A=Xa-mu;Bm=Xb-mu
        D=_d2(A,A); med=np.median(D[np.triu_indices(len(A),1)]); out=[]
        Dt=_d2(Bm,A)
        for C,g in params:
            K=np.exp(-g*D/med);Kt=np.exp(-g*Dt/med)
            s=SVR(kernel='precomputed',C=C,epsilon=0.05).fit(K,(ya-ym)/ys); out.append(s.predict(Kt)*ys+ym)
        return np.array(out)
    errs=np.zeros(len(grid))
    for tri,vai in inner_kfold(n,seed):
        pr=run(Xtr[tri],ytr[tri],Xtr[vai],grid); errs+=((pr-ytr[vai][None,:])**2).sum(1)
    j=int(np.argmin(errs)); return run(Xtr,ytr,Xte,[grid[j]])[0],dict(C=grid[j][0],g=grid[j][1])

from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
def _topk(Xa,ya,k):
    Xc=Xa-Xa.mean(0); yc=ya-ya.mean()
    r=np.abs((Xc*yc[:,None]).sum(0))/(np.sqrt((Xc**2).sum(0))*np.sqrt((yc**2).sum())+1e-12)
    return np.argsort(-r)[:k]
def _make_et(seed): return ExtraTreesRegressor(n_estimators=200,max_features=0.5,min_samples_leaf=1,random_state=seed,n_jobs=1)
def _make_xgb(seed): return xgb.XGBRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.5,reg_lambda=1.0,min_child_weight=1,random_state=seed,n_jobs=1,verbosity=0,tree_method='hist')
def fit_predict_tree(Xtr,ytr,Xte,seed,kind='et',ks=(20,60),return_model=False):
    n=len(ytr); mk=_make_et if kind=='et' else _make_xgb
    errs=np.zeros(len(ks))
    for tri,vai in inner_kfold(n,seed):
        for j,k in enumerate(ks):
            idx=_topk(Xtr[tri],ytr[tri],k); m=mk(seed).fit(Xtr[tri][:,idx],ytr[tri])
            errs[j]+=((m.predict(Xtr[vai][:,idx])-ytr[vai])**2).sum()
    j=int(np.argmin(errs)); k=ks[j]; idx=_topk(Xtr,ytr,k); m=mk(seed).fit(Xtr[:,idx],ytr)
    pred=m.predict(Xte[:,idx])
    if return_model: return pred,dict(k=k,idx=idx,model=m)
    return pred,dict(k=k,idx=idx)

# ---------- outer CV splits ----------
def outer_splits(y,R,K=6):
    lab=np.argsort(np.argsort(y,kind='stable'))//K  # 7 equal-size target-rank strata for K=6, n=42
    out=[]
    for r in range(R):
        skf=StratifiedKFold(K,shuffle=True,random_state=1000+r)
        out.append([(tr,te) for tr,te in skf.split(np.zeros(len(y)),lab)])
    return out

def metrics(y,p):
    e=p-y; ss=np.sum((y-y.mean())**2)
    return dict(R2=1-np.sum(e**2)/ss,RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(np.abs(e))),MAPE=float(np.mean(np.abs(e/y))*100),MedAE=float(np.median(np.abs(e))))
