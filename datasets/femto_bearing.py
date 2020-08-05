import numpy as np
import os
from collections import OrderedDict

import socket

KNOWN_HOSTS = ['marvin']
IN_COLAB = False
HOSTNAME = socket.gethostname()
DEFAULT_NPZ_FILE = None
ROOTFOLDER_PICKLES = None


try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True

    ROOTFOLDER_PICKLES = None  #'/Dataset/bearing_fatigue_dataset/'
    DEFAULT_NPZ_FILE ='/content/drive/My Drive/FEMTO_Bearings/first_stage_preproc_bearings.npz'
except:
    print("not running in colab.")
    if HOSTNAME not in KNOWN_HOSTS:
        print("Not supported! See/edit code.")

if HOSTNAME == 'marvin':
    ROOTFOLDER_PICKLES = '/mnt/860EVO_1TB/Dataset/bearing_fatigue_dataset/'
    DEFAULT_NPZ_FILE ='/mnt/860EVO_1TB/Dataset/bearing_fatigue_dataset/first_stage_preproc_bearings.npz'



class FEMTOBearingsDataset:
    """
    A class to manage the bearings dataset.
    """
    def __init__(self, rootfolder_pickles = ROOTFOLDER_PICKLES, npz_file = DEFAULT_NPZ_FILE, from_npz = True, filter_from_end_time = np.inf):

        self.npz_file = npz_file
        # The files may not be available. The npz is usually used (faster).
        self.all_files = ['AccB_1_6.pickle', 'AccB_2_2.pickle', 
                'AccB_2_5.pickle', 'AccB_2_4.pickle', 
                'AccB_2_6.pickle', 'AccB_1_1.pickle', 
                'AccB_2_7.pickle', 'AccB_1_5.pickle', 
                'AccB_3_2.pickle', 'AccB_2_1.pickle', 
                'AccB_3_1.pickle', 'AccB_2_3.pickle', 
                'AccB_3_3.pickle', 'AccB_1_7.pickle', 
                'AccB_1_2.pickle', 'AccB_1_4.pickle', 
                'AccB_1_3.pickle'] #[f for f in os.listdir(rootfolder_pickles) if 'pickle' in f]
        if from_npz:
            L = np.load(self.npz_file)
            self.yrem_s_raw, self.X, self.eid = [L[l] for l in L.files]
        else:
            self.rootfolder = rootfolder_pickles

            def load_bearings_dataset(all_files = self.all_files):
                fnames = all_files
                exp_ids_tot = [];
                yrem_tot = [];
                sensor_values_tot = []

                for kk,fname in enumerate(fnames):
                    p1 = pd.read_pickle(os.path.join(self.rootfolder_pickles,"bearing_fatigue_dataset",fname))
                    p1['abs_timestamp'] = (p1['h']*60+p1['m'])*60+p1['s']+p1['se-6']/1e6

                    sensors = p1.groupby("block").apply(lambda x : [x['abs_timestamp'].mean(),np.vstack([x['h_acc'].values, x['v_acc'].values]).T])
                    sens_val = np.dstack([v[1] for v in sensors.values])
                    sensor_values = np.swapaxes(np.swapaxes(sens_val,0,-1),1,2).astype('float32')
                    print(sensor_values.shape)

                    yrem = np.vstack([v[0] for v in sensors.values]).astype('float32')
                    yrem = np.abs(yrem - np.max(yrem))

                    exp_id = np.hstack([kk for v in range(yrem.shape[0])])

                    sensor_values_tot.append(sensor_values)
                    exp_ids_tot.append(exp_id);
                    yrem_tot.append(yrem)

                    print(kk)
                yrem_tot = np.vstack(yrem_tot).flatten()
                eid = np.hstack(exp_ids_tot).flatten()
                X__ = np.vstack(sensor_values_tot)
                return yrem_tot, eid, X__


            self.yrem_s_raw, self.X, self.eid = load_bearings_dataset()

        inds_filter = (self.yrem_s_raw < filter_from_end_time)
        self.yrem_s_raw = self.yrem_s_raw[inds_filter]
        self.X = self.X[inds_filter]
        self.eid = self.eid[inds_filter]

        eid_oh = np.zeros([self.eid.shape[0],np.max(self.eid) + 1])
        for k in np.unique(self.eid):
            eid_oh[self.eid == k,k] = 1.;

        
        self.eid_oh = eid_oh

        self.inds_exp_target,self.inds_exp_source = self.get_data_bearings_RUL()# call once with default parameters so the corresponding fields exist for further use.


    def get_dataset_config(self):
        return {'training_set' : self.inds_exp_source, 'validation_set' : self.inds_exp_target } 



    def get_data_bearings_RUL(self,min_samples_keep = 910,
                      normalization_factor_time = 25000,
                      yrem_norm_thresh = 0.0001):
        # min_samples_keep:           minimum number of samples an experiment needs to have to
        #                             be used in either training or test set.
        #
        # normalization_factor_time:  (manually selected) normalization factor for the remaining time variable
        #
        # yrem_norm_thresh:           also for numerical reasons - a threshold on the smallest value the
        #                             (normalized) output can take


        # normalization_factor_time = 25000#25000; # Manually selected normalization factor
        yrem_norm = self.yrem_s_raw/normalization_factor_time;
        self.normalization_factor_time = normalization_factor_time

        yrem_norm[yrem_norm<yrem_norm_thresh] = yrem_norm_thresh # For numerical reasons - otherwise NaNs occur.
        self.yrem_norm = yrem_norm

        conds_flag_source = [0,0,0,0,   1,1, 1,1,  2,2]
        inds_exp_source =  [4,6,8,15,  9,10,0,7,  2,11]

        conds_flag_target = [0,0,0,   1, 1, 1,  2]
        inds_exp_target  = [1,5,16,  12,13,14,  3]

        conds_flag = [*conds_flag_source, *conds_flag_target]
        inds_experiments = [*inds_exp_source, *inds_exp_target]
        exp_to_cond_dict = {k:v for k,v in zip(inds_experiments, conds_flag)}
        self.exp_to_cond_dict = exp_to_cond_dict
        loading_cond = [exp_to_cond_dict[k] for i,k in enumerate(np.argmax(self.eid_oh,1))]
        loading_oh = np.zeros([self.eid_oh.shape[0],3])
        for k in np.unique(loading_cond):
            loading_oh[loading_cond == k,k] = 1;
        self.loading_oh = loading_oh

        inds_experiments = [*inds_exp_source, *inds_exp_target]

        #############################################################
        #Remove experiments with too few entries

        ids_filter = [i for i,k in enumerate(np.argmax(self.eid_oh,1)) if k in inds_experiments]

        self.ids_filter = ids_filter

        exp_and_counts = [(kk,np.sum(self.eid_oh[:,kk] == 1)) for kk in range(len(inds_experiments))]
        inds_exp_source_new = [];
        for e,i in exp_and_counts:
            if i>min_samples_keep:
                inds_exp_source_new.append(e)

        inds_exp_source = inds_exp_source_new

        inds_exp_target = [16,2] # manually selected.
        for yy in inds_exp_target:
            inds_exp_source.remove(yy)

        counts_dict = OrderedDict(exp_and_counts);
        print("training set:")
        for k,m in [(i,exp_to_cond_dict[i]) for i in inds_exp_source]:
            print("%02i %i %s %i"%(k,m, self.all_files[k][5:8],counts_dict[k]))

        print("\ntesting set:")
        for k,m in [(i,exp_to_cond_dict[i]) for i in inds_exp_target]:
            print("%02i %i %s %i"%(k,m, self.all_files[k][5:8],counts_dict[k]))

        assert(len(set(inds_exp_source).intersection(inds_exp_target)) == 0)
        return inds_exp_target, inds_exp_source


