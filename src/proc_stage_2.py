import numpy as np
import pandas as pd
import os
import sys
import scipy as sp
from scipy.interpolate import interp1d
import time

def resample_and_featurize_dataset(dat_fname,nresample = 400,chunksize = 100000, nchunks = np.inf, root_dir = "."):
    """

    2nd pre-processing stage:
    2.1) Data from each cycle is separately re-sampled in order to yield consistent number of samples per cycle.
    2.2) The resampled data are saved to disk in order to make more economic use of memory, and the processing happens in chunks.

    The rationale behind the "feature" engineering is the following:

    *  Interpolation happens using the recorded timestamp for a fixed number
       of points. The controller seems to use different speed for different loading ranges.

    *  Since we have the timestamp, and because the load controller does not have
       constant rate, rates of change of displacement are used.

    *  An intuitive quantity that we see changing is the tangent stiffness. A quantity related
       to stiffness for each timepoint is computed.
       A custom formula is used, to accomodate the large variations
       in displacement ranges for different experiments and for different
       parts of the lifetime of each experiment, without explicitly mean-normalizing,
       since we cannot compute reliably means with so few experiments, and
       hopefully without losing potentially useful information. If future experiments 
       are available, these functions of time-dependent quantities are to be re-evaluated
       in terms of whether they are leaking or not explicit instance-bound information.
       
       Update 19/12/2019: Trained models seems to generalize relatively well without advanced domain 
                          independence regularization .Tried experiments with adversarial 
                          domain regularization. This regularization may have been the cause of that.

       Update 10/01/2020  Domain adaptation turns out to help (in  top-3 accuracy) for 3/4 cases (surely for 2/4 with same failure mode).

                          * The one that doesn't improve with adv.training is the one with the worst generalization performance 
                            when not using adversarial training. The same one is the one having the most complex failure surface and 
                            fails on the threads! 

                          * There seems to be a mistake in the labeling of the anchors - there is a damaged point 
                            in the nut consistent with the damage I see on the anchor. Document accompanying the dataset labes "Sb" for failure mode anchors 1 and 3. 
                            Consistency on failure mode for 1 and 3 is supported by the fact that when 1 or 3 are left-out, the prediction works for them (this is 
                            expected if they have simmilar failure modes and not different!)

    """
    fname = os.path.join(*[root_dir, dat_fname])
    save_append_file = os.path.join(*[root_dir, dat_fname + "_resampled"])

    start_fcn  = time.time()

    with open(save_append_file, 'a') as f:

        def proc_chunk(df_, save_append_file=save_append_file, nresample = 400):
            df_.groupby("Zyklen").apply(lambda x : x)
            df_.drop(axis = 1, labels = [c for c in df_.columns if 'Unnamed' in c])
            df_ = resample_cycles(df_, nresample)
            
            df_.to_csv(f, header=f.tell()==0)

        def resample_cycles(data, nresample):
            """
            Resample the dataset to a number of points per cycle.
            Useful for linear analyses and regression with non-convolutional simple FFDNNs.
            """
            def app_fun(x):
                t = x['Zeit[s]'].values
                # a small value from the beg. and end of the interpolation bounds
                # is used in order to avoid NaNs:
                tp = np.linspace(t.min()+1e-5, t.max()-1e-5, nresample, endpoint = True);
                dt = tp[1] - tp[0];

                intp = interp1d(t,x[['Weg[mm]','Kraft[kN]','Acceleration[g]']].values.T)
                res = pd.DataFrame(intp(tp).T,columns = ['WegQ','KraftQ','AccelQ'])
                res['MaxK'] = x['Kraft[kN]'].max();
                res['MinK'] = x['Kraft[kN]'].min();
                cc1 = np.convolve([-0.5,0.5],res['KraftQ'],'same')/dt;
                cc1[0] = cc1[1]; cc1[-1] = cc1[-2];
                res['Kdot'] = cc1;

                cc2 = np.convolve([-0.5,0.5],res['WegQ'],'same')/dt;
                cc2[0] = cc2[1]; cc2[-1] = cc2[-2];
                res['Wdot'] = cc2
                return res

            #app_func = resample_cycle(nresample)

            dfq_i =data.reset_index().groupby('Zyklen').apply(app_fun)
            return dfq_i


        df_partial_for_next = None

        times = []
        cycles=[]

        for i,df in enumerate(pd.read_csv(fname, chunksize=chunksize)):
            last_cyc = np.sort(df.Zyklen.unique())[-1]

            df_wo_partial = df[df['Zyklen'] != last_cyc]

            if df_partial_for_next is None:
                df_toproc = df_wo_partial
            else:
                df_toproc = pd.concat([df_partial_for_next, df_wo_partial])

            df_partial_for_next = df[df['Zyklen'] == last_cyc]
            # Get the last cycle to append it as partian - just in case..

            proc_chunk(df_toproc)
            times.append(time.time() - start_fcn )
            cycles.append(last_cyc)

            print("Cycles/s: {}".format(cycles[-1]/times[-1]))
            if i>nchunks:
                break

    return {"times":times, "cycles" : cycles}

if __name__ == "__main__":
    fname = sys.argv[1]
    root_dir = '../csv/subs_200k_200/subs_csvs/'
    resample_and_featurize_dataset(fname,chunksize = 100000, nchunks = np.inf, root_dir = root_dir)   

