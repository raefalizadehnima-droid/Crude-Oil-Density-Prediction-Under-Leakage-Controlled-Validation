from extra_common import *
R=3; splits=outer_splits(y,R); out={}
# E4 noise injection at test time (models trained on clean training data)
def sd_ref(b): 
    X=d[b] if b!='MS' else np.sqrt(d['MS']); return float(np.median(X.std(0)))
SD={b:sd_ref(b) for b in PP}; out['noise_ref_sd']=SD
levels=[0,0.01,0.03,0.1,0.3]; res={}
rng=np.random.default_rng(7)
for name,bl in (('FTIR',['FTIR']),('MS',['MS']),('ALL',ORDER)):
    for f in levels:
        oofs=[]
        for r in range(R):
            oof=np.zeros(n)
            for fi,(tr,te) in enumerate(splits[r]):
                B=make_blocks(tr)
                Xtr=early(B,tr,bl)
                # build noisy test blocks from raw inputs
                Xte=[]
                for b in bl:
                    raw=d[b][te].copy() if b!='MS' else np.sqrt(d['MS'][te])
                    raw=raw+rng.normal(0,f*SD[b],raw.shape)
                    if b=='MS': Pb=raw
                    else: Pb=prep_spectrum(raw,PP[b][0],PP[b][3])
                    # apply fitted block transform manually
                    blk=B[b]; Xt=Pb
                    if blk.msc: Xt=blk._msc(Xt)
                    Xte.append(((Xt-blk.mu)/blk.sd)/blk.fro)
                p,_=fit_predict_pls(Xtr,y[tr],np.hstack(Xte),seed=r*10+fi); oof[te]=p
            oofs.append(oof)
        res[f'{name}|{f}']=agg(oofs); print('NOISE',name,f,res[f'{name}|{f}']['RMSE'][0],flush=True)
out['noise']=res
# E5 VIP-based variable selection inside folds (PLS; analogue of earlier chemometric practice)
vip={}
for name,bl in (('MS',['MS']),('FTIR',['FTIR']),('ALL',ORDER)):
    oofs=[];kept=[]
    for r in range(R):
        oof=np.zeros(n)
        for fi,(tr,te) in enumerate(splits[r]):
            B=make_blocks(tr); Xtr=early(B,tr,bl);Xte=early(B,te,bl)
            p,info=fit_predict_pls(Xtr,y[tr],Xte,seed=r*10+fi,return_model=True)[0:2] if False else (None,None)
            pr,inf=fit_predict_pls(Xtr,y[tr],Xte,seed=r*10+fi,return_model=True)
            m=inf['model'];k=inf['k'];v=pls_vip(m,k); sel=np.where(v>1)[0]
            if len(sel)<5: sel=np.argsort(-v)[:5]
            p2,_=fit_predict_pls(Xtr[:,sel],y[tr],Xte[:,sel],seed=r*10+fi); oof[te]=p2; kept.append(len(sel)/Xtr.shape[1])
        oofs.append(oof)
    vip[name]=dict(metrics=agg(oofs),frac_kept_mean=float(np.mean(kept)),frac_kept_sd=float(np.std(kept)))
    print('VIP',name,vip[name]['metrics']['RMSE'],vip[name]['frac_kept_mean'],flush=True)
out['vip']=vip
# E6 LOOCV for PLS/Ridge
loo={}
for name,bl in (('FTIR',['FTIR']),('MS',['MS']),('ALL',ORDER)):
    for mn,fn in (('PLS',fit_predict_pls),('Ridge',fit_predict_ridge)):
        oof=np.zeros(n)
        for i in range(n):
            tr=np.delete(np.arange(n),i);te=np.array([i]); B=make_blocks(tr)
            p,_=fn(early(B,tr,bl),y[tr],early(B,te,bl),seed=i); oof[i]=p[0]
        loo[f'{name}|{mn}']=metrics(y,oof); print('LOO',name,mn,loo[f'{name}|{mn}']['RMSE'],flush=True)
out['loocv']=loo
json.dump(out,open('out/extra2.json','w'),default=float)
