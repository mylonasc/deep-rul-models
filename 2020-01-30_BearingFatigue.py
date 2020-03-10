#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('ls bearing_fatigue_dataset/')


# ## Adversarial learning on the NASA accelerated fatigue bearing dataset
# The dataset is associated with a challenge for the PHM conference on 2012.
# The motivation of this work is to investigate if and when adversarial learning on raw time-series performs well. The hypotheses of this work are the following:
# 
# * **Dilated DenseNets** are flexible enough to work with raw time-series (this is not an established fact in SHM applications)
# * Adversarial learning can be used to normalize implicitly for **domain shifts** without manual intervention. 
# 
# The following need to be performed:
# * Preprocessing (making the dataset appropriate for training with the adversarial densenet - domain labels etc)
# * Train/test splitting for the raw accelerometer time-series
# * write the optimization loop
# * run prelim. experiments - hopefully something will fit. 
# * write code for performing 1-vs-rest experiments to test generalization.

# In[213]:


import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as pplot
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Flatten, BatchNormalization, Dropout

import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)
tf.keras.backend.set_session(sess)
from tensorflow.math import confusion_matrix


# In[214]:


from src.dilated_densenet_adversarial import make_model, make_densenet_parametrization_dictionary


# In[254]:


from src.util import plot_confusion_matrix
from src.experiments_utils import *

ndn = 5
ndn2 = 5

experiment_parametrization = {
    'dilation_rates_dn1': [2,4,8,16,32,64,128,512,8,16,32,64,128,512],#,4,8,16,32],
    'kernel_sizes_dn1' :  [10] * ndn,#,5,5],
    'nfilters_dn1' :  [20]*ndn,
    'dilation_rates_dn2' : [4]*ndn2,
    'kernel_sizes_dn2' : [5]*ndn2,
    'nfilters_dn2': [10]*ndn2
}

make_network_description_experiments_A(**experiment_parametrization)
model_json = make_network_json_experiments_A(**experiment_parametrization)
#
model_json["classifier"]["network_layers"][-1]['params']['units'] = 10
model_json["densenet_1d"]["network_layers"][0]["params"]["Variational"] = False
model_json["densenet_1d"]["network_layers"] = [
    model_json["densenet_1d"]["network_layers"][0],model_json["densenet_1d"]["network_layers"][-1]
]
model_json['inputs'][0]['shape'] = (2559,2)

# Make the model act as a regressor:
model_json['classifier']['network_layers'][-1]['params']['units'] = 1
model = make_model(model_json)


# In[258]:


#model.summary()
l = model.get_layer("DN_DN1_Out")


# In[265]:


## A helper function to return the pooling layer (the feature detector output) of a network:
get_global_pool_layer = lambda mm : mm.get_layer([l.name for l in mm.layers if 'global_average' in l.name][0])


# In[62]:


import pandas as pd
import matplotlib.pyplot as pplot
import numpy as np
p1 = pd.read_pickle("bearing_fatigue_dataset/AccB_1_1.pickle")


# In[246]:


def load_bearings_dataset():
    fnames = ['AccB_1_1.pickle','AccB_1_2.pickle','AccB_2_1.pickle','AccB_2_2.pickle','AccB_3_1.pickle','AccB_3_2.pickle']
    exp_ids_tot = [];
    yrem_tot = [];
    sensor_values_tot = []

    for kk,fname in enumerate(fnames):
        p1 = pd.read_pickle("bearing_fatigue_dataset/%s"%fname)
        p1['abs_timestamp'] = p1['h']+p1['m']*60+p1['s']+p1['se-6']/1e6
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
    X = np.vstack(sensor_values_tot)
    return yrem_tot, eid, X
yrem, eid, X = load_bearings_dataset()


# ## Train-test splitting:

# In[249]:


eid_oh = np.ones(eid.shape[0],np.max(eid) + 1)
for k in np.unique(eid):
    eid_oh[eid == k] = 1.;


# In[253]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, d_train, d_test = train_test_split(X, yrem, eid_oh, stratify = eid_oh)


# ## Training

# In[266]:


feature_extractor = keras.Model(inputs = model.inputs, outputs = get_global_pool_layer(model).output)


# In[273]:


#feature_extractor(x_train[0:10]).eval()


# In[275]:



epochs = 100; #training_parameters["nepochs"]
batchsize =100;

from tensorflow.math import confusion_matrix
import tensorflow.keras as keras
from tensorflow.keras import layers 

LRATE_MAX = 0.01;
BURNIN_EPOCHS = 10;

c_loss =0.0;
CLOSS_MAX = 0.1;

es = keras.callbacks.EarlyStopping(monitor='val_Y_output_loss', mode='min', verbose=1, patience=10, restore_best_weights = True)
model_opt = keras.optimizers.Adam(learning_rate=0.0);



class Top3AccStrong(tf.keras.callbacks.Callback):
    def __init__(self):
        self.top3acc_strong = []
    def on_epoch_end(self,epoch,logs = None):
        top3acc, cmatrix_strong = plot_cmatrix_strong_return_top3acc(leave_exp_out, epoch, 0.001,root_folder = "/tmp")
        self.top3acc_strong.append(top3acc)

class plot_PCA(tf.keras.callbacks.Callback):
    def __init__(self):
        None
    def on_epoch_end(self,epoch,logs = None):
        a,b = get_vv_vals()
        pp = PCA(n_components=2).fit_transform((a - np.mean(a,0)) /np.std(a))
        pplot.scatter(pp[:,0], pp[:,1],c = b)
        pplot.show()
        #top3acc = plot_cmatrix_strong_return_top3acc(leave_exp_out, epoch, 0.001,root_folder = "/tmp")

class BurnIn(tf.keras.callbacks.Callback):
    def __init__(self, burnin_epochs = None, lrate_max = LRATE_MAX):
        self.learning_rate = lrate_max
        self.burnin_epochs = burnin_epochs
        
    def on_epoch_end(self,epoch, logs = None):
        if epoch <= self.burnin_epochs:
            de = epoch/self.burnin_epochs
            model_opt.learning_rate = de * self.learning_rate
        else:
            None

def top3_acc(labels, logits):
    return keras.metrics.top_k_categorical_accuracy(labels,logits, k=3)


def plot_confusion():
    cmat = confusion_matrix(np.argmax(Yoh,1),np.argmax(Yhat,1))
    pplot.pcolor(cmat.eval())
    pplot.show()


loss_fcn = {"Y_output" :lambda y,yhat : tf.reduce_mean(tf.pow(y-yhat,2)),
            "ExpID" : lambda y,yhat : tf.compat.v1.losses.softmax_cross_entropy(y,yhat),
           }

loss_w = {"Y_output": 1.,"ExpID" : 0.1}

# GOAL: Minimize the useless discriminator while maximizing the useful classifier.

# useful discriminator:
model.compile(optimizer =  model_opt,
              loss =loss_fcn,
              loss_weights = loss_w)

train_in, train_out = [{"timeseries_input": x_train},{"Y_output":y_train,
                                                     "ExpID" : d_train}]
test_in, test_out = [{"timeseries_input": x_test},{"Y_output":y_test,
                                                     "ExpID" : d_test}]
# I use the left-out experiment as validation set. 
# The only info used from the left-out set is when to stop training.
# Results are good also by using a validation set from the "seen" experiments.
#test_in, test_out = [{"timeseries_input" : Xstrong}, {"Y_output" : YstrongOH, "ExpID" : np.zeros([YstrongOH.shape[0],3])}]
burnin = BurnIn(burnin_epochs = BURNIN_EPOCHS, lrate_max = LRATE_MAX)
history_d = model.fit(train_in, train_out, epochs=epochs,
                     batch_size=batchsize,
                      validation_data = [test_in, test_out],# "Rc_output" :  RemCTest}],
                     callbacks=[es, burnin])#, Top3AccStrong()])#, InspectActivations()]);#, Top3AccStrong()])



# In[277]:



pplot.plot(history_d.history['loss'])
pplot.show()
pplot.savefig("losses")


# In[ ]:




