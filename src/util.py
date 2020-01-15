import pandas as pd
import matplotlib.pyplot as pplot
import scipy as sp
from scipy.interpolate import interp1d
import pickle
import numpy as np

def digitize(dat, start_num, end_num, nbins):
    """
    Digitize continuous column wrt number of bins.
    """
    dd = np.linspace(start_num, end_num,  nbins+1)
    bins_start = dd[0:-1]
    bins_end = dd[1:]
    q_vector = []
    for s,e  in zip(bins_start, bins_end):
        q_vector.append((dat>=s) * (dat<e))

    return np.vstack(q_vector).T*1

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          figsize = (10,10),
                          dpi = 300,
                          cmap=None,
                          normalize=False, flip_horiz=True, subtitle_y = None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize, dpi = dpi)


    if flip_horiz:
        cm = np.fliplr(cm)


    plt.pcolor(cm,cmap = cmap)# interpolation='nearest', cmap=cmap)

    #plt.imshow(cm,interpolation='nearest', cmap=cmap)
    plt.title(title)


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        if flip_horiz:
            plt.xticks(np.flip(tick_marks)+0.5, target_names, rotation=45)
        else:
            plt.xticks(tick_marks+0.5, target_names, rotation=45)
        plt.yticks(tick_marks+0.5, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #plt.colorbar()
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j+0.5, i+0.5, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j+0.5, i+0.5, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True remaining cycles')
    plt.xlabel('Predicted remaining cycles\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()

#from src.util import digitize
def load_hilti_fatigue_data(keep_from_end = 150000,    n_cycles_per_sub_block = 5, n_resampled = 300,nclasses = 15, leave_exp_out = 'VA_2', stage_2_from_disk = False,user_normalization = None):

    #keep_from_end = 150000;
    #nclasses = 15
    if stage_2_from_disk:
        #keep_from_end = 300000;

        for ds_idx in [1,2,3,4]:
            print("proc:"+str(ds_idx))
            ds =pd.read_pickle("csv/subs_200k_200/subs_csvs/VA_test{}_stage_2.pickle".format(ds_idx))
            ds.reset_index(inplace = True)
            remc = (ds.Zyklen.max() - ds.Zyklen)

            ds = ds[remc<=keep_from_end]
            ds['RemCycles'] = remc
            ds['ExpId'] = "VA_%i"%ds_idx
            if ds_idx == 1:
                dataset = ds
            else:
                dataset = pd.concat([dataset,ds])


        def make_block_id_column(x):
            block_id = np.cumsum(np.hstack([True,np.diff(x.Zyklen.values)>1]))
            x['BlockId'] = block_id
            return x

        #n_resampled = 300 # uniform re-sampling per cycle.

        d = dataset.groupby("ExpId").apply(make_block_id_column)
    else:
        d = pd.read_csv("csv/subs_200k_200/subs_dataframe_{}k_from_end".format(int(keep_from_end/1000)))

    ######################################################################################################
    #n_resampled = 300

    def make_data_per_block(x):
        # This loses some cycles from the end.
        # This function should not perform normalization because there will be information leakage.
        # The only exception are accelerations. There is a weird shift in the accelerations which 
        # could be from static electricity or sth. It was decided to remove it here (just mean centering 
        # - this should not cause information leakage).
        n_stop_idx = x.shape[0] - (x.shape[0] % (n_resampled*n_cycles_per_sub_block))
        x = x[0:n_stop_idx]

        n_cycles_in_block = int(x.shape[0]/ n_resampled)
        shape_ = [n_cycles_in_block,int(x.shape[0]/n_cycles_in_block)]

        tangent_stiffness = x['KraftQ']/(x['WegQ']**0.15+ 0.1)
        tangent_stiffness = tangent_stiffness.values.reshape(shape_)
        f = x['KraftQ'].values.reshape(shape_)
        u = x['WegQ'].values.reshape(shape_)
        kdot = x['Kdot'].values.reshape(shape_)
        accel = x['AccelQ'].values.reshape(shape_)
        wdot = x['Wdot'].values.reshape(shape_)
        accel = accel - np.mean(accel)
        vals_dat = np.dstack([tangent_stiffness, f, u, accel, kdot, wdot])
        vals_dat = vals_dat.reshape([-1,n_cycles_per_sub_block*n_resampled,vals_dat.shape[-1]])

        get_scalar = lambda scname, x : x[scname].values.reshape(shape_).reshape([-1,n_cycles_per_sub_block*300])[:,0]

        rem_c_dat = get_scalar('RemCycles', x)
        #rem_c_dat = rem_c_dat.reshape([-1, n_cycles_per_sub_block])

        eid = get_scalar('ExpId',x)

        return vals_dat, rem_c_dat, eid

    training_data = d.groupby(["ExpId","BlockId"]).apply(make_data_per_block)


    X = np.vstack([v[0] for v in training_data.values])
    RemC = np.concatenate([v[1] for v in training_data.values])
    Eid = np.concatenate([v[2] for v in training_data.values])
    Yoh = digitize(RemC, -1, np.max(RemC)+1, nclasses)


    d.drop(axis = 1 , labels = [c for c in d.columns if 'Unnamed' in c], inplace = True)
    d.drop(axis = 1 , labels = [c for c in d.columns if 'level' in c], inplace = True)
    mean_disp = d.groupby(["ExpId",'BlockId']).apply(lambda x : x['WegQ'].max())
    w_exp_means = mean_disp.reset_index().groupby("ExpId").apply(lambda x : x.quantile(0.10))
    eids = ['VA_1','VA_2',"VA_3","VA_4"]
    for ee in eids:
        X[Eid == ee, :, 2] = X[Eid == ee,:,2] - w_exp_means.loc[ee].values[1]

    ## Filtering manually some outliers. This was based on manual inspection up to 100k cycles from end.
    f1 = np.min(X[:,:,0],1)>0.1
    f2 = np.max(np.abs(X[:,:,3]),1)<2
    f3 = np.max(X[:,:,4],1) < 1000
    f4 = np.max(np.abs(X[:,:,5]),1)<50
    ftot = f1 * f2 * f3 *f4



    X_clean = X[ftot,:,:]
    Y_clean = RemC[ftot]
    Eid_clean = Eid[ftot]
    Yoh_clean = Yoh[ftot]

    ts_sub    = np.mean(X_clean[:,:,0])
    ts_divide = np.max(X_clean[:,:,0])

    f_sub    = np.mean(X_clean[:,:,1])
    f_divide = np.max(X_clean[:,:,1])

    u_sub    = np.mean(X_clean[:,:,2])
    u_divide = np.max(X_clean[:,:,2])

    accel_sub =  np.mean(X_clean[:,:,3])
    accel_divide =  np.std(X_clean[:,:,3])

    kdot_sub = np.mean(X_clean[:,:,4])
    kdot_divide = np.std(X_clean[:,:,4])

    wd_sub = np.mean(X_clean[:,:,5])
    wd_divide = np.std(X_clean[:,:,5])


    normalization = np.array([[ts_sub, ts_divide],
                              [f_sub, f_divide],
                              [u_sub, u_divide],
                              [accel_sub, accel_divide],
                             [kdot_sub, kdot_divide],
                             [wd_sub, wd_divide]]).astype("float32")

    del d
    if user_normalization is not None:
        normalization = user_normalization

    Xnorm = (X_clean.astype("float32") - normalization[:,0])
    Xnorm = Xnorm/ normalization[:,1]

    eid_vector = Eid_clean

    
    Xstrong, Ystrong, YstrongOH, eid_vector_strong = [Q[Eid_clean == leave_exp_out] for Q in [Xnorm,Y_clean,Yoh_clean,eid_vector]]
    X, Y,Yoh, eid_vector = [Q[Eid_clean !=  leave_exp_out] for Q in [Xnorm,Y_clean,Yoh_clean,eid_vector]]

    return {"training_instances" : [X, Y, Yoh, eid_vector] , "validation_instances" : [Xstrong, Ystrong, YstrongOH, eid_vector_strong], "normalization" : normalization, "nclasses" : nclasses}





class FatigueCyclesDataset(object):
    """
    This is sort of "deprecated". It turns out that it does some 
    trivial preproc that is better treated with some small scripts and functions. 
    Left here for future reference.
    """
    def __init__(self,load_raw_from = "All_experiments" ,
                 load_object_from_file = None,
                 n_resample_cycles = 400,
                 nquant_levels = 20,
                 QZ = 5000,
                 NC_fromEnd = 1000):
        """
        Manipulation of a sub-sampled dataset for transforming it in a form 
        appropriate for training predictive models
        
        Parameters:
            load_raw_from     : A file containing a dataframe (with appropriate columns)
            
            load_object_from  : In case the preprocessing this class performs has already been saved to a file, it is possible
                                to load that file directly.
                                
            n_resample_cycles : Cycles have different number of samples. 
                                In order to allow for algorithms that can work on fixed number of inputs the dataset is 
                                resampled to a fixed length. The loading rate information is lost, but may be integrated
                                explicitly in the analysis through post-proc. of other parameters
                                
            nquant_levels     : The quantization levels in case of classification-type dataset (task is classify acc. to remaining cycles)
            
            QZ                : To the remaining cycles, first I apply np.floor(C/QZ) * QZ in order to make quantization more reliable (fewer levels).
                                A good value for this is the value the sub-sampled dataset was subsampled according to.
                                IMPORTANT: If too many quantization levels occur, or if quantization levels are
                                different for different experiments change so that is not the case!
                
            NC_fromEnd        : How many cycles from failure to keep. This further truncates the dataset to make it more manageable, but also has 
                                an experiment-related bias removal role (stratification). I.e: if substantially more samples
                                to failure are available for a speciffic experiment, say, 3 million vs 5 thousand, the training 
                                will focus on the experiment with more samples and ignore the rest.
                         
        """

        if load_object_from_file is not None:
            self.load_from_file(load_object_from_file)
            self.INIT_FROM_FILE = True
            
        else:
            self.NC_fromEnd = NC_fromEnd;
            self.n_resample_cycles = n_resample_cycles;
            self.use_datasets = ['VA_test1' ,'VA_test2','VA_test3','VA_test4']

            self.n_resample_cycles = n_resample_cycles
            self.QZ = QZ;
            self.subsampled_raw_loaded_from = load_raw_from
            self.raw_subsampled = self.load_raw_subsampled_dataset(load_raw_from)

            self.resampled_dataset = self.resample_cycles(nresample = self.n_resample_cycles, 
                                                          use_datasets = self.use_datasets)

            # further subsampling the dataset to focus it closer to failure (and balance wrt the different number of samples in diff. experiments)
            self.resampled_dataset = [dd.tail(self.NC_fromEnd * self.n_resample_cycles) for dd in self.resampled_dataset]
            self.nquant_levels = nquant_levels
            keep_rows_from_end = self.NC_fromEnd * self.n_resample_cycles;
            self.quantized_rem_cycles_columns  = self.make_quantized_rem_cycles(self.nquant_levels,
                                                                     self.QZ,
                                                                     keep_rows_from_end)

            self.INIT_FROM_FILE = False
    
    def load_raw_subsampled_dataset(self,load_from, keep_subsampled = True):
        """
        load_from       : the file that contains an aggregated pandas dataframe, that holds a set of sub-sampled data.
                          the sub-sampling happens in another python function that processes the raw textfiles in chunks.
                    
        keep_subsampled : If the loaded sub-sampled dataframes are to be kept loaded in RAM. 
                          If low on RAM memory set to `False`.
        """
        all_exp = pd.read_pickle("All_experiments")
        all_exp = all_exp.set_index(['Experiment','Zyklen','S/No'])
        return all_exp
        
    def resample_cycles(self, nresample, use_datasets):
        """
        Resample the dataset to a number of points per cycle.
        Useful for linear analyses and regression with non-convolutional simple FFDNNs.
        """
        def app_fun(x):
            t = x['Zeit[s]'].values
            # a small value from the beg. and end of the interpolation bounds
            # is used in order to avoid NaNs:
            tp = np.linspace(t.min()+1e-5, t.max()-1e-5, nresample, endpoint = True); 
            intp = interp1d(t,x[['Weg[mm]','Kraft[kN]','Accel[g]']].values.T)
            maxK = x['Kraft[kN]'].max()
            res = pd.DataFrame(intp(tp).T,columns = ['WegQ','KraftQ','AccelQ'])
            res['MaxK'] = maxK;
            return res
    
        #app_func = resample_cycle(nresample)

        dfq_i = [self.raw_subsampled.loc[kk].reset_index().groupby('Zyklen').apply(app_fun) for kk in use_datasets]
        return dfq_i

    
    def make_quantized_rem_cycles(self, nlevels, Qrange, keep_rows_from_end):
        """
        nlevels          : How many quantization classes to be returned
        NC_fromEnd       : The number of cycles from end to use (and quantize)
        Qrange           : Something related to size of quantization bins (has to do with previous pre-proc as well. Avoid changing)
        nresample_cycles : the constant number of samples per cycle the dataset has been resampled to.
        """
        
        for dd in self.resampled_dataset:
            dd.reset_index(inplace = True)
            dd['RemZyklen'] = dd['Zyklen'].max() - dd['Zyklen']
            dd['RemZyklenQuantized'] = (np.floor(dd['RemZyklen'].values/Qrange)*Qrange)
        
        vv = [pd.np.digitize(dd.RemZyklenQuantized , bins = np.linspace(0,keep_rows_from_end,nlevels)) for dd in self.resampled_dataset]

        return vv
    
    def write_to_file(self, filename):
        """
        Saves the object to a file
        """
        with open(filename,'wb') as f:
            pickle.dump(self, f)
            
    def load_from_file(self,filename):
        self = pickle.Unpickler(open(filename,'rb'))
