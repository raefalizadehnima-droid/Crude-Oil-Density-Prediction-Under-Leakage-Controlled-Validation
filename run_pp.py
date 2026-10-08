import numpy as np, json, time, sys
import lib
from lib import *
d=get_data(); y=d['y']; R=10
splits=outer_splits(y,R)
def variants():
    v=[]
    for b,w in (('NIR',21),('FTIR',11)):
        for k,msc,sc in (('raw',False,'center'),('snv',False,'center'),('msc',True,'center'),('sg1',False,'center'),('snv_sg1',False,'center'),('sg2',False,'center')):
            v.append((b,k,msc,sc,w))
    for k,sc in (('raw','center'),('raw','auto'),('sqrt','center'),('log','center'),('log','auto'),('rel_log','center')):
        v.append(('MS',k,False,sc,0))
    return v
res={}
for (b,k,msc,sc,w) in variants():
    t0=time.time()
    X=prep_ms(d[b],k) if b=='MS' else prep_spectrum(d[b],k,w)
    blk=Block(X,msc=msc,scale=sc)
    for mname,fn in (('PLS',fit_predict_pls),('Ridge',fit_predict_ridge)):
        oof=np.zeros((R,len(y))); ks=[]
        for r in range(R):
            for fi,(tr,te) in enumerate(splits[r]):
                blk.fit(tr); p,info=fn(blk.transform(tr),y[tr],blk.transform(te),seed=r*10+fi)
                oof[r,te]=p; ks.append(info.get('k',info.get('rel_alpha')))
        m=[metrics(y,oof[r]) for r in range(R)]
        res[f'{b}|{k}|{sc}|{mname}']=dict(R2=float(np.mean([a['R2'] for a in m])),R2sd=float(np.std([a['R2'] for a in m])),RMSE=float(np.mean([a['RMSE'] for a in m])),RMSEsd=float(np.std([a['RMSE'] for a in m])),MAE=float(np.mean([a['MAE'] for a in m])),k_med=float(np.median(ks)))
        print(f'{b:5s}{k:8s}{sc:7s}{mname:6s} R2={res[f"{b}|{k}|{sc}|{mname}"]["R2"]:.3f} RMSE={res[f"{b}|{k}|{sc}|{mname}"]["RMSE"]:.4f} k~{np.median(ks)}',flush=True)
    print('  t=%.0fs'%(time.time()-t0),flush=True)
json.dump(res,open('out/pp_comparison.json','w'),indent=1)
