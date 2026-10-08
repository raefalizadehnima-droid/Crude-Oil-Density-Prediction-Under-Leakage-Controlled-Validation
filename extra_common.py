import numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
from lib import *; from fusion import *
d=get_data(); y=d['y']; mz=d['mz']; n=len(y)
PP={'NIR':('msc',True,'center',21),'FTIR':('snv_sg1',False,'center',11),'MS':('sqrt',False,'center',0)}
def raw_prep(b,X=None):
    X=d[b] if X is None else X
    return prep_spectrum(X,PP[b][0],PP[b][3]) if b!='MS' else prep_ms(X,'sqrt')
RAW={b:raw_prep(b) for b in PP}
def make_blocks(tr,RAWd=None):
    RAWd=RAWd or RAW
    B={b:Block(RAWd[b],msc=PP[b][1],scale=PP[b][2]).fit(tr) for b in PP}; return B
ORDER=['FTIR','MS','NIR']
def early(B,idx,bl,w=None,scaled=True):
    w=w or {}
    return np.hstack([B[b].transform(idx,scaled)*w.get(b,1.0) for b in bl])
def agg(oof_list,yy=None):
    yy=y if yy is None else yy
    M=[metrics(yy,o) for o in oof_list]
    return {k:(float(np.mean([m[k] for m in M])),float(np.std([m[k] for m in M],ddof=1) if len(M)>1 else 0)) for k in M[0]}
