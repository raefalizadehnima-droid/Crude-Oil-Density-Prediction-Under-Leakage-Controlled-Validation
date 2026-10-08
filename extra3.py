from extra_common import *
import pickle
R=3; splits=outer_splits(y,R); out={}
# ---------- E7: preprocessing chosen inside the inner loop ----------
VAR={'FTIR':[('raw',False,'center'),('snv',False,'center'),('msc',True,'center'),('sg1',False,'center'),('snv_sg1',False,'center'),('sg2',False,'center')],
     'MS':[('raw',False,'center'),('raw',False,'auto'),('sqrt',False,'center'),('log',False,'center'),('log',False,'auto')]}
def build(b,v):
    k,msc,sc=v
    X=prep_ms(d['MS'],k) if b=='MS' else prep_spectrum(d[b],k,11)
    return Block(X,msc=msc,scale=sc)
e7={}
for b in ('FTIR','MS'):
    VB=[build(b,v) for v in VAR[b]]; oofs=[];sel=[]
    for r in range(R):
        oof=np.zeros(n)
        for fi,(tr,te) in enumerate(splits[r]):
            errs=[]
            for blk in VB:
                e=0
                for tri,vai in inner_kfold(len(tr),r*10+fi):
                    a,c=tr[tri],tr[vai]; blk.fit(a); p,_=fit_predict_pls(blk.transform(a),y[a],blk.transform(c),seed=fi)
                    e+=np.sum((p-y[c])**2)
                errs.append(e)
            j=int(np.argmin(errs)); sel.append(j); blk=VB[j].fit(tr)
            p,_=fit_predict_pls(blk.transform(tr),y[tr],blk.transform(te),seed=r*10+fi); oof[te]=p
        oofs.append(oof)
    e7[b]=dict(metrics=agg(oofs),selected_counts={str(VAR[b][j]):int(sel.count(j)) for j in range(len(VAR[b]))})
    print('E7',b,e7[b],flush=True)
out['preproc_nested']=e7
# ---------- E8: block permutation importance + coefficient stability (ALL early-fusion PLS) ----------
rng=np.random.default_rng(11); Q=30
base=[np.zeros(n) for _ in range(R)]; perm={b:[[np.zeros(n) for _ in range(R)] for q in range(Q)] for b in ORDER}
coefs=[];ks=[]
for r in range(R):
    for fi,(tr,te) in enumerate(splits[r]):
        B=make_blocks(tr); Xtr=early(B,tr,ORDER);Xte=early(B,te,ORDER)
        p,inf=fit_predict_pls(Xtr,y[tr],Xte,seed=r*10+fi,return_model=True)
        m=inf['model'];k=inf['k'];beta=m['B'][k-1]; coefs.append(beta.astype(np.float32)); ks.append(k)
        base[r][te]=p
        sizes=[B[b].X.shape[1] for b in ORDER]; off=np.cumsum([0]+sizes)
        for bi,b in enumerate(ORDER):
            for q in range(Q):
                Xp=Xte.copy(); pi=rng.permutation(len(te)); Xp[:,off[bi]:off[bi+1]]=Xte[pi][:,off[bi]:off[bi+1]]
                perm[b][q][r][te]=(Xp-inf['mu'])@beta+inf['ym']
    print('E8 repeat',r,flush=True)
rb=np.mean([metrics(y,base[r])['RMSE'] for r in range(R)])
imp={}
for b in ORDER:
    dd=[np.mean([metrics(y,perm[b][q][r])['RMSE'] for r in range(R)])-rb for q in range(Q)]
    imp[b]=dict(mean=float(np.mean(dd)),sd=float(np.std(dd,ddof=1)))
out['block_perm_importance']=dict(base_RMSE=float(rb),imp=imp)
C=np.array(coefs); sizes=[RAW[b].shape[1] for b in ORDER]; off=np.cumsum([0]+sizes); stab={}
for bi,b in enumerate(ORDER):
    S=np.corrcoef(C[:,off[bi]:off[bi+1]]); iu=np.triu_indices(len(C),1); stab[b]=dict(mean=float(S[iu].mean()),min=float(S[iu].min()))
out['coef_stability']=stab; out['ncomp_selected_ALL_PLS']=ks
# ---------- final models on all 42 (interpretation only; NOT used for performance) ----------
allidx=np.arange(n); B=make_blocks(allidx); X=early(B,allidx,ORDER)
p,inf=fit_predict_pls(X,y,X,seed=0,return_model=True); m=inf['model'];k=inf['k'];beta=m['B'][k-1]; vipv=pls_vip(m,k)
Xc=X-inf['mu']; phi=Xc*beta[None,:]       # per-sample feature contributions (linear SHAP for PLS; exact)
out['final_pls']=dict(k=int(k),train_R2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)))
share={}
for bi,b in enumerate(ORDER):
    c=phi[:,off[bi]:off[bi+1]].sum(1); share[b]=dict(var_share=float(np.cov(c,y)[0,1]/np.var(y,ddof=1)))
out['final_pls']['block_cov_share']=share
np.savez('out/interp_pls.npz',beta=beta,vip=vipv,phi_abs=np.abs(phi).mean(0),phi=phi.astype(np.float32),off=off,wn_nir=d['wn_nir'],wn_ftir=d['wn_ftir'],mz=mz,coefs=C)
# SHAP for XGBoost on pooled features of ALL
import shap
from lib import _topk,_make_xgb
def pool_idx(nn,w): return np.arange(nn)//w
pool={'NIR':pool_idx(d['NIR'].shape[1],40),'FTIR':pool_idx(d['FTIR'].shape[1],10),'MS':np.floor((mz-mz.min())/5).astype(int)}
axes={'NIR':d['wn_nir'],'FTIR':d['wn_ftir'],'MS':mz}
P=[];names=[]
for b in ORDER:
    g=pool[b]; Xb=B[b].transform(allidx)
    for j in range(g.max()+1):
        s=g==j
        P.append(Xb[:,s].mean(1)); names.append(f'{b}@{axes[b][s].mean():.1f}')
P=np.array(P).T
res={}
for kk in (60,):
    idx=_topk(P,y,kk); mdl=_make_xgb(0).fit(P[:,idx],y)
    ex=shap.TreeExplainer(mdl); sv=ex.shap_values(P[:,idx])
    np.savez('out/interp_xgb.npz',sv=sv,X=P[:,idx],names=np.array(names)[idx],idx=idx,expected=ex.expected_value)
json.dump(out,open('out/extra3.json','w'),default=float)
print('done')
