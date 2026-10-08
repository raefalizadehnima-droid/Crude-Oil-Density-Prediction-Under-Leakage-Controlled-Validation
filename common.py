import pandas as pd, numpy as np
P='/mnt/user-data/uploads/E_Gjelsvik_2023_density_oil_'
def load():
    y=pd.read_csv(P+'y_target.csv',index_col=0)['y_target']
    N=pd.read_csv(P+'NIR_data.csv',index_col=0);F=pd.read_csv(P+'FTIR_data.csv',index_col=0);M=pd.read_csv(P+'APPI-FTICR_MS_data.csv',index_col=0)
    return y,N,F,M
