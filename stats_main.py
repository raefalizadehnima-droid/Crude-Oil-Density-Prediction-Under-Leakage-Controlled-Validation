import numpy as np, pandas as pd, json
from scipy import stats
from lib import *
d=get_data(); y=d['y']; n=len(y)
Z=np.load('out/oof_main_part0.npz'); R=Z[Z.files[0]].shape[0]
splits=outer_splits(y,R)
rng=np.random.default_rng(SEED)
def m_all(P): 
    M=[metrics(y,P[r]) for r in range(R)]; return {k:(np.mean([m[k] for m in M]),np.std([m[k] for m in M],ddof=1)) for k in M[0]}
B=2000; boots=rng.integers(0,n,(B,n))
def boot_ci(P,key):
    vals=[]
    for b in boots:
        yy=y[b]; ss=np.sum((yy-yy.mean())**2)
        e=P[:,b]-yy[None,:]
        vals.append(np.mean(1-np.sum(e**2,1)/ss) if key=='R2' else np.mean(np.sqrt(np.mean(e**2,1))))
    return np.percentile(vals,[2.5,97.5])
rows=[]
for k in Z.files:
    P=Z[k]; assert not np.isnan(P).any(),k
    m=m_all(P); fs,mo=k.split('|')
    rows.append(dict(featureset=fs,model=mo,R2=m['R2'][0],R2_sd=m['R2'][1],RMSE=m['RMSE'][0],RMSE_sd=m['RMSE'][1],MAE=m['MAE'][0],MAPE=m['MAPE'][0],MedAE=m['MedAE'][0],
        R2_lo=boot_ci(P,'R2')[0],R2_hi=boot_ci(P,'R2')[1],RMSE_lo=boot_ci(P,'RMSE')[0],RMSE_hi=boot_ci(P,'RMSE')[1]))
T=pd.DataFrame(rows).sort_values('RMSE'); T.to_csv('out/table_performance.csv',index=False)
pd.set_option('display.width',250); print(T.round(4).to_string(index=False))
# paired comparison function
def fold_mse(P):
    out=[]
    for r in range(R):
        for tr,te in splits[r]: out.append(np.mean((P[r,te]-y[te])**2))
    return np.array(out)
def compare(a,b):
    Pa,Pb=Z[a],Z[b]
    ea=((Pa-y)**2).mean(0); eb=((Pb-y)**2).mean(0)    # per-sample MSE averaged over repeats
    dr=[np.sqrt(ea[i].mean())-np.sqrt(eb[i].mean()) for i in boots]
    w=stats.wilcoxon(ea,eb).pvalue
    fa,fb=fold_mse(Pa),fold_mse(Pb); diff=fa-fb; J=len(diff); nt,ntr=7,35
    t=diff.mean()/np.sqrt((1/J+nt/ntr)*diff.var(ddof=1)); p=2*stats.t.sf(abs(t),J-1)
    return dict(A=a,B=b,dRMSE=np.sqrt(ea.mean())-np.sqrt(eb.mean()),ci_lo=np.percentile(dr,2.5),ci_hi=np.percentile(dr,97.5),p_wilcoxon=w,p_NB_corrected_t=p)
json.dump(rows,open('out/table_performance.json','w'),default=float)

fam=[('ALL|PLS','FTIR|PLS'),('ALL|PLS','MS|PLS'),('ALL|PLS','NIR|PLS'),('NIR+FTIR|PLS','FTIR|PLS'),('MS+FTIR|PLS','FTIR|PLS'),
     ('ALL|SO-PLS','FTIR|PLS'),('ALL|Late-stack','FTIR|PLS'),('MS+FTIR|SO-PLS','FTIR|PLS'),('NIR+FTIR|Ridge','FTIR|Ridge'),('ALL|Ridge','FTIR|Ridge'),
     ('FTIR|SVR','FTIR|PLS'),('FTIR|SVR','FTIR|Ridge'),('FTIR|SVR','FTIR|ET'),('FTIR|SVR','FTIR|XGB'),('FTIR|SVR','baseline|MeanPredictor'),
     ('FTIR|SVR','NIR+FTIR|Ridge'),('FTIR|SVR','ALL|SO-PLS')]
C=pd.DataFrame([compare(a,b) for a,b in fam])
for col in ('p_wilcoxon','p_NB_corrected_t'):
    p=C[col].values; o=np.argsort(p); adj=np.empty(len(p)); run=0
    for rank,i in enumerate(o):
        run=max(run,min(1,(len(p)-rank)*p[i])); adj[i]=run
    C[col+'_holm']=adj
C.to_csv('out/table_comparisons.csv',index=False); print(C.round(4).to_string(index=False))
# error analysis for key models: sample-level mean OOF residuals
res={}
for k in ('FTIR|SVR','FTIR|PLS','ALL|SO-PLS','ALL|PLS','MS|Ridge','NIR+FTIR|Ridge'):
    P=Z[k].mean(0); e=P-y; res[k]=dict(bias=float(e.mean()),sd=float(e.std(ddof=1)),slope=float(np.polyfit(y,P,1)[0]),intercept=float(np.polyfit(y,P,1)[1]),
      q50=float(np.quantile(np.abs(e),.5)),q80=float(np.quantile(np.abs(e),.8)),q90=float(np.quantile(np.abs(e),.9)),q95=float(np.quantile(np.abs(e),.95)),
      within0p01=float((np.abs(e)<=0.01).mean()),within0p02=float((np.abs(e)<=0.02).mean()),worst=[(int(i+1),float(y[i]),float(P[i])) for i in np.argsort(-np.abs(e))[:4]],
      shapiro_p=float(stats.shapiro(e).pvalue), spearman_absres_y=float(stats.spearmanr(np.abs(e),y)[0]))
json.dump(res,open('out/error_analysis.json','w'),indent=1,default=float); print(json.dumps(res,indent=0,default=float)[:2500])
