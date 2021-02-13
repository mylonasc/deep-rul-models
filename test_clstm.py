
from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Sequential,Model
import tensorflow.keras as keras

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)

import numpy as np
import matplotlib.pyplot as pplot
tfd = tfp.distributions
from ipywidgets import FloatSlider, IntSlider, interact, interactive


import pandas as pd
from datasets.femto_bearing import FEMTOBearingsDataset
from utils import get_graph_data
from graphnet_utils import GraphNetFunctionFactory

femto_dataset = FEMTOBearingsDataset()

learning_set = ['1_2','1_3','1_4','1_5','2_1','2_5','2_6','3_3'] # Giving decent results.
#learning_set = ['1_3','1_2','1_4','1_5','2_1','2_5','3_2','3_3'] # Giving decent results.
# learning_set = ['1_3','1_2','1_4','1_5','1_7','2_1','2_5','2_6','3_3'] # Giving decent results.
# learning_set = ['1_3','1_2','1_4','1_5',      '2_1','2_5','2_6','3_3'] # Giving decent results.
# learning_set = ['1_3','1_2','1_4','1_5','2_2','2_1','2_5','2_6','3_3'] # Giving decent results.
#learning_set = ['1_3','1_1','1_4','1_5','2_3','2_5','3_3'] # Giving decent results.

#learning_set.extend([femto_dataset.file_suffix[k_] for k_ in [3,4,5,,10]])
print(learning_set)
learning_set_inds = [femto_dataset.file_suffix.index(k) for k in learning_set]
eval_inds = [f for f in femto_dataset.file_suffix if f not in learning_set]

val_set_inds = [femto_dataset.file_suffix.index(k) for k in eval_inds]
APPLY_DEFAULT_SETUP = True
if APPLY_DEFAULT_SETUP:
    femto_dataset.inds_exp_source = learning_set_inds
    femto_dataset.inds_exp_target = val_set_inds

gn = GraphNetFunctionFactory()
make_conv_input_head_node_function = lambda edge_size , **kwargs: gn.make_conv_input_head_node_function(edge_size, **kwargs)
make_node_function = lambda *args, **kwargs : gn.make_node_function(*args,**kwargs)
make_edge_function_gi = lambda *args , **kwargs : gn.make_edge_function_gi
make_gamma_node_observation_mlp = lambda *args,**kwargs : gn.make_gamma_node_observation_mlp(*args , **kwargs)

from src.convlstm import Conv1DLSTMCell

from src.convlstm import make_layer,make_seq


import src.convlstm as clstm
ksize = 3;
nfilts = 18;
nfilts2 = 50

p=[{"type"  : "input"    , "kwargs" : {"shape" : (None,2)}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : 1,    "filters" : nfilts2,    "strides" : 1, "use_bias" : False, "name" : "conv_fcnA"}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2, "use_bias" : False, "name" : "conv_fcnB"}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2,"activation":"relu", "use_bias" : False, "name" : "conv_fcnC"}},
 {"type" : "dropout"   ,"kwargs":{"rate" : 0.2}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,    "filters" : nfilts2,    "strides" : 2, "use_bias" : False, "name" : "conv_fcnD"}},
 {"type" : "avg_pool1d", "kwargs":{"pool_size" : 2}},

#  {"type" : "conv1d"    ,"kwargs":{"kernel_size" : 1,    "filters" : nfilts2,    "strides" : 1, "use_bias" : False, "name" : "conv_fcnA2"}},
#  {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2, "use_bias" : False, "name" : "conv_fcnB2"}},
#  {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2,"activation":"relu", "use_bias" : False, "name" : "conv_fcnC2"}},
#  {"type" : "dropout"   ,"kwargs":{"rate" : 0.2}},
#  {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,    "filters" : nfilts2,    "strides" : 2, "use_bias" : False, "name" : "conv_fcnD2"}},
#  {"type" : "avg_pool1d", "kwargs":{"pool_size" : 2}},

 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : 1,    "filters" : nfilts2,    "strides" : 1, "use_bias" : False, "name" : "conv_fcnA3"}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2, "use_bias" : False, "name" : "conv_fcnB3"}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,"filters" : nfilts, "strides" : 2,"activation":"relu", "use_bias" : False, "name" : "conv_fcnC3"}},
 {"type" : "dropout"   ,"kwargs":{"rate" : 0.2}},
 {"type" : "conv1d"    ,"kwargs":{"kernel_size" : ksize,    "filters" : nfilts2,    "strides" : 2, "use_bias" : False, "name" : "conv_fcnD3"}},
 {"type" : "global_average_pooling", "kwargs":{}},
 {"type" : "dense" , "kwargs" : {"units" : 20,"use_bias" : False}}]

head_dt = [{"type" : "input", "kwargs" : {"shape" : (1,)}},
                   {"type" : "dense" , "kwargs" : {"units" : 20, "use_bias" : False}}];







make_seq(p, name = "name")

cell = Conv1DLSTMCell(30, dt_model_params=head_dt, cnn_model_params=p)
cell.build((None,1), (None,None,2))
rnn = tf.keras.layers.RNN(cell)

x_in = Input(shape = (None,1))
x_in_ts = Input( shape = (None,None, 2))
y_out = rnn((x_in,x_in_ts))
gamma_out_mlp = make_gamma_node_observation_mlp(n_node_state_output)
y_out = gamma_out_mlp(y_out)
rnn_gamma_model = Model(inputs = [x_in, x_in_ts] , outputs = y_out)

