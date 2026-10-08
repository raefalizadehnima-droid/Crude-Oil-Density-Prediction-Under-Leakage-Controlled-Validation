from extra_common import *
out={}
R=3; splits=outer_splits(y,R)
# E1 label permutation (nested pipeline unchanged) for FTIR-PLS and ALL early PLS, one repeat per permutation
def run_cv(yv,spl,bl,fn=fit_predict_pls,w=None):
    oof=np.zeros(n)
    for fi,(tr,te) in enumerate(spl):
        B=make_blocks(tr); p,_=fn(early(B,tr,bl,w),yv[tr],early(B,te,bl,w),seed=fi); oof[te]=p
    return oof
rng=np.random.default_rng(SEED); t0=time.time()
for name,bl in (('FTIR',['FTIR']),('ALL',ORDER)):
    obs=[metrics(y,run_cv(y,outer_splits(y,1)[0],bl))['R2']] 
    null=[]
    for q in range(30):
        yp=rng.permutation(y); null.append(metrics(yp,run_cv(yp,outer_splits(yp,1)[0],bl))['R2'])
    out['perm_'+name]=dict(obs_R2_rep0=obs[0],null_R2=null,p=(1+sum(v>=obs[0] for v in null))/(1+len(null)))
    print(name,'perm done',obs[0],np.mean(null),np.max(null),time.time()-t0,flush=True)
# E2 learning curve
fr=[0.4,0.6,0.8,1.0]; lc={}
for name,bl in (('FTIR',['FTIR']),('MS',['MS']),('ALL',ORDER)):
    for mn,fn in (('PLS',fit_predict_pls),('Ridge',fit_predict_ridge)):
        for f in fr:
            oofs=[]
            for r in range(R):
                oof=np.zeros(n); rr=np.random.default_rng(100+r)
                for fi,(tr,te) in enumerate(splits[r]):
                    sub=np.sort(rr.choice(tr,int(round(f*len(tr))),replace=False))
                    B=make_blocks(sub); p,_=fn(early(B,sub,bl),y[sub],early(B,te,bl),seed=fi); oof[te]=p
                oofs.append(oof)
            lc[f'{name}|{mn}|{f}']=agg(oofs)
            print('LC',name,mn,f,lc[f'{name}|{mn}|{f}']['RMSE'][0],flush=True)
out['learning_curve']=lc
# E3 block-weight sensitivity (early-fusion PLS, ALL)
bw={}
for blk in ('MS','NIR'):
    for wv in (0.25,0.5,1,2,4):
        oofs=[]
        for r in range(R):
            oof=np.zeros(n)
            for fi,(tr,te) in enumerate(splits[r]):
                B=make_blocks(tr); w={blk:wv}; p,_=fit_predict_pls(early(B,tr,ORDER,w),y[tr],early(B,te,ORDER,w),seed=r*10+fi); oof[te]=p
            oofs.append(oof)
        bw[f'{blk}|{wv}']=agg(oofs); print('BW',blk,wv,bw[f'{blk}|{wv}']['RMSE'][0],flush=True)
out['block_weight']=bw
json.dump(out,open('out/extra1.json','w'),default=float)
