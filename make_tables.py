import json,numpy as np,pandas as pd,collections
from PIL import Image
T=pd.read_csv('out/table_performance.csv');C=pd.read_csv('out/table_comparisons.csv')
pp=json.load(open('out/pp_comparison.json'));err=json.load(open('out/error_analysis.json'));e3=json.load(open('out/extra3.json'));e1=json.load(open('out/extra1.json'));e2=json.load(open('out/extra2.json'))
I=json.load(open('out/info_main_part0.json'))
f=lambda x,d=3:f'{x:.{d}f}'
out={}
def row(r): return [r.featureset,r.model,f'{r.R2:.3f} ± {r.R2_sd:.3f}',f'{r.R2_lo:.2f}–{r.R2_hi:.2f}',f'{r.RMSE:.4f}',f'{r.RMSE_lo:.4f}–{r.RMSE_hi:.4f}',f'{r.MAE:.4f}',f'{r.MAPE:.2f}',f'{r.MedAE:.4f}']
sel=[('FTIR','SVR'),('FTIR','Ridge'),('FTIR','PLS'),('NIR+FTIR','Ridge'),('ALL','SO-PLS'),('ALL','Late-stack'),('FTIR','ET'),('FTIR','XGB'),('MS+FTIR','SO-PLS'),('ALL','Ridge'),('ALL','PLS'),('MS+FTIR','PLS'),('MS','ENet'),('MS','PLS'),('MSdesc','ENet'),('MS+NIR','SO-PLS'),('NIR','Ridge'),('NIR','PLS'),('baseline','MeanPredictor')]
out['perf']=[row(T[(T.featureset==a)&(T.model==b)].iloc[0]) for a,b in sel]
def cell(fs,mo):
    r=T[(T.featureset==fs)&(T.model==mo)]
    return '–' if len(r)==0 else f'{r.iloc[0].R2:.3f} ({r.iloc[0].RMSE:.4f})'
out['multi']=[[fs]+[cell(fs,m) for m in ('PLS','Ridge','ENet','SVR','ET','XGB','Late-mean','Late-stack','SO-PLS')] for fs in ['NIR','FTIR','MS','MSdesc','NIR+FTIR','MS+NIR','MS+FTIR','ALL']]
out['comp']=[[r.A,r.B,f'{r.dRMSE:+.4f}',f'{r.ci_lo:+.4f} to {r.ci_hi:+.4f}',f'{r.p_wilcoxon:.4f}',f'{r.p_wilcoxon_holm:.3f}',f'{r.p_NB_corrected_t:.3f}',f'{r.p_NB_corrected_t_holm:.3f}'] for _,r in C.iterrows()]
ek=['FTIR|SVR','FTIR|PLS','NIR+FTIR|Ridge','ALL|SO-PLS','ALL|PLS','MS|Ridge']
out['err']=[[k.replace('|',' / '),f"{err[k]['bias']:+.4f}",f"{err[k]['q50']:.4f}",f"{err[k]['q90']:.4f}",f"{err[k]['q95']:.4f}",f"{err[k]['within0p01']*100:.0f}",f"{err[k]['within0p02']*100:.0f}",f"{err[k]['slope']:.2f}",', '.join(str(w[0]) for w in err[k]['worst'][:3])] for k in ek]
# preprocessing table
rows=[]
for k,v in pp.items():
    b,p,sc,m=k.split('|')
    if m=='PLS':
        r=pp[k.replace('PLS','Ridge')]; rows.append([b,f'{p} ({sc})',f"{v['R2']:.3f} ± {v['R2sd']:.3f}",f"{v['RMSE']:.4f}",f"{r['R2']:.3f} ± {r['R2sd']:.3f}",f"{r['RMSE']:.4f}"])
out['pp']=rows
# hyper
H={}
for k,v in I.items():
    fs,mo=k.split('|')
    if mo=='PLS': H[k]='%g (%d–%d)'%(np.median([i['k'] for i in v]),min(i['k'] for i in v),max(i['k'] for i in v))
    elif mo=='Ridge': H[k]='%.0e (%d/30 at grid min.)'%(np.median([i['rel_alpha'] for i in v]),sum(i['rel_alpha']<=1.01e-8 for i in v))
    elif mo=='ENet': H[k]='%g non-zero'%np.median([i['nnz'] for i in v])
    elif mo=='SVR': c=collections.Counter((i['C'],i['g']) for i in v).most_common(1)[0]; H[k]='C=%g, γ×%g (%d/30)'%(c[0][0],c[0][1],c[1])
    elif mo in('ET','XGB'): c=collections.Counter(i['k'] for i in v).most_common(1)[0]; H[k]='k=%d (%d/30)'%(c[0],c[1])
    elif mo=='SO-PLS': H[k]='comps '+str([int(x) for x in np.median([i['comps'] for i in v],axis=0)])
    elif mo=='Late-stack': H[k]='w='+str([round(float(x),2) for x in np.mean([i['w'] for i in v],axis=0)])
out['hyp']=[[fs]+[H.get(f'{fs}|{m}','–') for m in ('PLS','Ridge','ENet','SVR','ET','XGB')] for fs in ['NIR','FTIR','MS','NIR+FTIR','MS+FTIR','ALL']]
out['hyp_fus']=[[fs,H.get(f'{fs}|SO-PLS','–'),H.get(f'{fs}|Late-stack','–')] for fs in ['NIR+FTIR','MS+NIR','MS+FTIR','ALL']]
# regions
I2=np.load('out/interp_pls.npz');off=I2['off'];pa=I2['phi_abs']
def reg(ax,s,lst):
    o=[]
    for name,lo,hi in lst:
        m=(ax>=lo)&(ax<hi); o.append([name,f'{lo}–{hi}',int(m.sum()),f'{m.mean()*100:.1f}',f'{s[m].sum()/s.sum()*100:.1f}'])
    return o
out['reg_ftir']=reg(I2['wn_ftir'],pa[off[0]:off[1]],[('O–H/N–H region',3100,3600),('Aromatic/olefinic =C–H stretch',3000,3100),('Aliphatic C–H stretch',2800,3000),('Baseline/overtone region',1700,2800),('Aromatic C=C region',1550,1700),('CH2/CH3 bending',1400,1550),('CH3 symmetric bending',1330,1400),('Fingerprint',900,1330),('Aromatic C–H out-of-plane',800,900),('High-wavenumber tail',3600,4000)])
out['reg_nir']=reg(I2['wn_nir'],pa[off[2]:],[('Combination region',4000,4800),('4800–5400',4800,5400),('First-overtone region',5400,6200),('6200–7000',6200,7000),('7000–8000',7000,8000),('Second-overtone region',8000,9000),('9000–9600',9000,9600)])
mzv=I2['mz'];sm=pa[off[1]:off[2]];out['reg_ms']=[[f'{lo}–{lo+100}',int(((mzv>=lo)&(mzv<lo+100)).sum()),f'{((mzv>=lo)&(mzv<lo+100)).mean()*100:.1f}',f'{sm[(mzv>=lo)&(mzv<lo+100)].sum()/sm.sum()*100:.1f}'] for lo in range(100,1100,100)]
# robustness
lc=e1['learning_curve'];bw=e1['block_weight'];nz=e2['noise']
out['rob_lc']=[[nm+' '+mn]+[f"{lc[f'{nm}|{mn}|{fr}']['RMSE'][0]:.4f}" for fr in (0.4,0.6,0.8,1.0)] for nm,mn in (('FTIR','PLS'),('FTIR','Ridge'),('MS','PLS'),('MS','Ridge'),('ALL','PLS'),('ALL','Ridge'))]
out['rob_bw']=[[b]+[f"{bw[f'{b}|{w}']['RMSE'][0]:.4f}" for w in (0.25,0.5,1,2,4)] for b in ('MS','NIR')]
out['rob_noise']=[[nm]+[f"{nz[f'{nm}|{l}']['RMSE'][0]:.4f}" for l in (0,0.01,0.03,0.1,0.3)] for nm in ('FTIR','MS','ALL')]
out['rob_vip']=[[k,f"{v['metrics']['R2'][0]:.3f}",f"{v['metrics']['RMSE'][0]:.4f}",f"{v['frac_kept_mean']*100:.1f}"] for k,v in e2['vip'].items()]
out['rob_loo']=[[k.replace('|',' / '),f"{v['R2']:.3f}",f"{v['RMSE']:.4f}",f"{v['MAE']:.4f}"] for k,v in e2['loocv'].items()]
pe=e3['preproc_nested'];out['rob_pp']=[[b,f"{v['metrics']['R2'][0]:.3f} ± {v['metrics']['R2'][1]:.3f}",f"{v['metrics']['RMSE'][0]:.4f}"] for b,v in pe.items()]
out['perm']={k:dict(obs=e1[k]['obs_R2_rep0'],mean=float(np.mean(e1[k]['null_R2'])),mx=float(np.max(e1[k]['null_R2'])),p=e1[k]['p']) for k in ('perm_FTIR','perm_ALL')}
# image sizes
import glob;out['img']={p.split('/')[-1]:Image.open(p).size for p in glob.glob('out/figs/*.png')}
json.dump(out,open('out/tables.json','w'),ensure_ascii=False,indent=1)
print({k:(len(v) if hasattr(v,'__len__') else v) for k,v in out.items()}); print(out['img'])
