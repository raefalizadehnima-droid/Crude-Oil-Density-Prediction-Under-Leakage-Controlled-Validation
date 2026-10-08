import numpy as np, json, time, sys, pickle
from lib import *; from fusion import *
d=get_data(); y=d['y']; mz=d['mz']
R=int(sys.argv[1]); r_start=int(sys.argv[2]) if len(sys.argv)>2 else 0
splits=outer_splits(y,R)
PP={'NIR':('msc',True,'center',21),'FTIR':('snv_sg1',False,'center',11),'MS':('sqrt',False,'center',0)}
RAW={}
RAW['NIR']=prep_spectrum(d['NIR'],'msc',21); RAW['FTIR']=prep_spectrum(d['FTIR'],'snv_sg1',11); RAW['MS']=prep_ms(d['MS'],'sqrt')
DESC,desc_names=ms_descriptors(d['MS'],mz)
# pooling (stateless) for tree models
def pool_idx(n,w): return np.arange(n)//w
pool={'NIR':pool_idx(d['NIR'].shape[1],40),'FTIR':pool_idx(d['FTIR'].shape[1],10),'MS':np.floor((mz-mz.min())/5).astype(int)}
def pooled(X,key):
    g=pool[key]; ng=g.max()+1; out=np.zeros((X.shape[0],ng)); cnt=np.bincount(g,minlength=ng)
    for j in range(ng): out[:,j]=X[:,g==j].sum(1)/cnt[j]
    return out
FSETS={'NIR':['NIR'],'FTIR':['FTIR'],'MS':['MS'],'NIR+FTIR':['FTIR','NIR'],'MS+NIR':['MS','NIR'],'MS+FTIR':['FTIR','MS'],'ALL':['FTIR','MS','NIR']}
blocks={k:Block(RAW[k],msc=PP[k][1],scale=PP[k][2]) for k in RAW}
descblk=Block(DESC,False,'auto')
OOF={};INFO={}
def put(key,r,te,p): OOF.setdefault(key,np.full((R,len(y)),np.nan))[r,te]=p
def log(key,info): INFO.setdefault(key,[]).append({k:v for k,v in info.items() if k not in('idx','model')})
t0=time.time()
for r in range(r_start,R):
    for fi,(tr,te) in enumerate(splits[r]):
        seed=r*10+fi
        for b in blocks.values(): b.fit(tr)
        descblk.fit(tr)
        Z={k:(b.transform(tr,True),b.transform(te,True)) for k,b in blocks.items()}   # block-scaled
        Zraw={k:(b.transform(tr),b.transform(te)) for k,b in blocks.items()}           # unscaled for pooling
        Pl={k:(pooled(Zraw[k][0],k),pooled(Zraw[k][1],k)) for k in Zraw}
        put('baseline|MeanPredictor',r,te,np.full(len(te),y[tr].mean()))
        # engineered MS descriptors (unimodal)
        Xd=(descblk.transform(tr),descblk.transform(te))
        for mn,fn in (('PLS',fit_predict_pls),('Ridge',fit_predict_ridge),('ENet',fit_predict_enet),('SVR',fit_predict_svr)):
            p,i=fn(Xd[0],y[tr],Xd[1],seed); put(f'MSdesc|{mn}',r,te,p); log(f'MSdesc|{mn}',i)
        for mn,kd in (('ET','et'),('XGB','xgb')):
            p,i=fit_predict_tree(Xd[0],y[tr],Xd[1],seed,kd,ks=(10,30)); put(f'MSdesc|{mn}',r,te,p); log(f'MSdesc|{mn}',i)
        for fs,bl in FSETS.items():
            Xtr=np.hstack([Z[b][0] for b in bl]);Xte=np.hstack([Z[b][1] for b in bl])
            for mn,fn in (('PLS',fit_predict_pls),('Ridge',fit_predict_ridge),('ENet',fit_predict_enet),('SVR',fit_predict_svr)):
                p,i=fn(Xtr,y[tr],Xte,seed); put(f'{fs}|{mn}',r,te,p); log(f'{fs}|{mn}',i)
            Ptr=np.hstack([Pl[b][0] for b in bl]);Pte=np.hstack([Pl[b][1] for b in bl])
            for mn,kd in (('ET','et'),('XGB','xgb')):
                p,i=fit_predict_tree(Ptr,y[tr],Pte,seed,kd,ks=(20,60)); put(f'{fs}|{mn}',r,te,p); log(f'{fs}|{mn}',i)
            if len(bl)>1:
                pm,ps,i=fit_predict_late([Z[b][0] for b in bl],y[tr],[Z[b][1] for b in bl],seed)
                put(f'{fs}|Late-mean',r,te,pm);put(f'{fs}|Late-stack',r,te,ps);log(f'{fs}|Late-stack',i)
                p,i=fit_predict_soplS([Zraw[b][0] for b in bl],y[tr],[Zraw[b][1] for b in bl],seed)
                put(f'{fs}|SO-PLS',r,te,p);log(f'{fs}|SO-PLS',i)
        print(f'r={r} fold={fi} elapsed={time.time()-t0:.0f}s',flush=True)
    np.savez(f'out/oof_main_part{r_start}.npz',**{k:v for k,v in OOF.items()}); json.dump(INFO,open(f'out/info_main_part{r_start}.json','w'),default=float)
print('done',flush=True)
