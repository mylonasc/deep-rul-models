import time
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import import interp1d
import os 
import sys


resample = 300
def app_fun(x):
    t = x['Zeit[s]'].values
    # a small value from the beg. and end of the interpolation bounds
    # is used in order to avoid NaNs:
    tp = np.linspace(t.min()+1e-5, t.max()-1e-5, nresample, endpoint = True);
    dt = tp[1] - tp[0];

    intp = interp1d(t,x[['Weg[mm]','Kraft[kN]','Acceleration[g]']].values.T)
    res = pd.DataFrame(intp(tp).T,columns = ['WegQ','KraftQ','AccelQ'])
    #res['MaxK'] = x['Kraft[kN]'].max();
    #res['MinK'] = x['Kraft[kN]'].min();
    cc1 = np.convolve([-0.5,0.5],res['KraftQ'],'same')/dt;
    cc1[0] = cc1[1]; cc1[-1] = cc1[-2];
    res['Kdot'] = cc1;

    cc2 = np.convolve([-0.5,0.5],res['WegQ'],'same')/dt;
    cc2[0] = cc2[1]; cc2[-1] = cc2[-2];
    res['Wdot'] = cc2
    return res

start = time.time()

for ii_ in range(3,5):
    print("processing: "+ str(ii_))
    
    proc_start = time.time()
    df = pd.read_csv("../csv/subs_200k_200/subs_csvs/VA_test{}.csv".format(ii_))

    for cc in ["Kraft[kN]", "Weg[mm]","Acceleration[g]"]:
        df[cc] = df[cc].astype("float32")
    df.drop(axis = 1, columns = ["System Date", "S/No"], inplace = True)

    df_res = df.reset_index().groupby("Zyklen").apply(app_fun)
    
    for cc in df_res.columns:
        df_res[cc] = df_res[cc].astype('float32')
        
    df_res.to_pickle("../csv/subs_200k_200/subs_csvs/VA_test{}_stage_2.pickle".format(ii_))
    proc_end = time.time()
    print("total time: {}".format( proc_end - start ))
    print("time for last dataframe:{}".format(proc_end - proc_start))
    del df_res
