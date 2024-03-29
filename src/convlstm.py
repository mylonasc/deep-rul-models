import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.compat.v1 import ConfigProto,InteractiveSession

# Buggy memory management in cuda 10.1+ workaround:
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, RNN, Dropout, Conv2D, AveragePooling1D, AveragePooling2D, MaxPool1D, MaxPool2D

import matplotlib.pyplot as pplot
import numpy as np



# When needing more layers, implement the interface here. 
#
def make_input(kwargs):
    #print(args,kwargs)
    return Input(**kwargs)

def make_conv1d(kwargs):
    return Conv1D(**kwargs)

def make_avg_pooling1d(kwargs):
    return AveragePooling1D(**kwargs)

def make_avg_pooling2d(kwargs):
    return AveragePooling2D(**kwargs)

def make_max_pooling1d(kwargs):
    return AveragePooling2D(**kwargs)

def make_max_pooling2d(kwargs):
    return AveragePooling2D(**kwargs)

def make_conv2d(kwargs):
    return Conv2D(**kwargs)

def make_dropout(kwargs):
    return Dropout(**kwargs)

def make_ga_pooling(dummy,**dummy2):
    return  tf.keras.layers.GlobalAveragePooling1D()

def make_dense(kwargs):
    return Dense(**kwargs)

def make_layer(layer_param):
    """
    Like sequential model but allowing 
    to define composite layers easily 
    and parametrized by a single json.
    """
    type_to_fn = {"conv1d" : make_conv1d,
                  "conv2d":make_conv2d,
                  "avg_pool1d":make_avg_pooling1d,
                  "avg_pool2d":make_avg_pooling2d,
                  "max_pool1d":make_max_pooling1d,
                  "max_pool2d":make_max_pooling2d,
                  "dropout" : make_dropout,
                  "global_average_pooling" : make_ga_pooling,
                 "input" : make_input,
                 "dense" : make_dense}

    return type_to_fn[layer_param["type"]](layer_param['kwargs'])


def make_seq(layers_data, name ):
    
    s = tf.keras.Sequential(name = name)
    for l in layers_data:
        print(l)
        s.add(make_layer(l))
    return s


default_cnn_layer_params = [{"type" : "input", "kwargs" : {"shape" : (None,2)}},
                 {"type" : "conv1d", "kwargs" : {"kernel_size" : 3,"filters":10}},
                 {"type" :"dropout" , "kwargs" : {"rate": 0.2}},
                 {"type" : "conv1d", "kwargs" : {"kernel_size" : 3,"filters":10}},
                 {"type" : "global_average_pooling" , "kwargs": {} }]

default_head_dt = [{"type" : "input", "kwargs" : {"shape" : (1,)}},
                   {"type" : "dense" , "kwargs" : {"units" : 10, "use_bias" : False}}];
        
class Conv1DLSTMCell(keras.layers.LSTMCell):
    """
    A 1D ConvLSTM Cell with parametrized transitions.
    """
    def __init__(self, units, dt_model_params = default_head_dt, 
            cnn_model_params = default_cnn_layer_params, 
            concat_size = None, conv_factory = None, dt_factory = None, **kwargs):
        """
        Create a conv1D/RNN cell - treats larger multi-scale sequences more easilly than either LSTM or CNN.
        Practical notes:
        * To care less about scaling of weights/gradients etc, try and have the outputs of the cnn_model and the dt_model the same dimensionality (otherwise scaling issues will arize).

        parmaeters:
            units: a global size parameter
            dt_model_params  : parameters for the model that casts whatever it is that parametrizes transition between RNN cell observations.
            cnn_model_params : parameters for the model that acts as a feature extractor for instantaneous observations.
        
        optional keyword parameters:
            conv_factory     : overrides the default conv head construction 
            dt_factory       : overrides the default transition parametrization head extractor

          
        """
        
        self.use_provided_conv_factory = False  ; # may be overriden
        self.use_provided_dt_factory = False ;# may be overriden

        # Attention: These are overriden if a factory is provided.
        self.dt_model_params = dt_model_params
        self.cnn_model_params = cnn_model_params

        # Having these factory methods makes it easier to compare architectures (I can use the exact same factories for different) 
        if conv_factory is not None: 
            self.conv_head_factory = conv_factory
            self.use_provided_conv_factory = True

        if dt_factory is not None:
            self.dt_factory = dt_factory
            self.use_provided_dt_factory = True

        
        super(Conv1DLSTMCell, self).__init__( units,**kwargs)
        
    def build(self,other_state_input_shape,timeseries_input_shape):
        """
        parameters:
          other_state_input_shape  : A conditioning input (not passed to the CNN)
          timeseries_input_shape   : A timeseries (high-freq.) input that is passed to a CNN
        """
        
        if self.use_provided_conv_factory == False:
            self.h_cnn = make_seq(self.cnn_model_params, name = "cnn")
        else:
            self.h_cnn = self.conv_head_factory(name = "cnn")

        self.h_cnn.build(input_shape = timeseries_input_shape)

        if not self.use_provided_dt_factory:
            self.h_x_dt = make_seq(self.dt_model_params, name = "transition_parametrization")
        else:
            self.h_x_dt = self.conv_head_factory(name = "transition_parametrization")


        self.h_x_dt.build(input_shape = other_state_input_shape)

        
        self.concat_size  = self.h_x_dt.output_shape[-1] + self.h_cnn.output_shape[-1] #self.h_cnn.output.shape[-1] + self.h_x_dt.output.shape[-1]
        #c = tf.keras.layers.concatenate([self.h_x_dt.output, self.h_cnn.output])
        super(Conv1DLSTMCell, self).build(input_shape = (None,self.concat_size))
        
    def call(self, inputs, states, **kwargs):
        l1,l2 = [self.h_x_dt(inputs[0]),self.h_cnn(inputs[1])]
        inputs_  = tf.keras.layers.concatenate([l1,l2])
        return super(Conv1DLSTMCell, self).call(inputs_, states,**kwargs)
        
        
if __name__ == "__main__":
    
    cell = Conv1DLSTMCell(10, dt_model_params=default_head_dt, cnn_model_params=default_cnn_layer_params)
    cell.build((None,1), (None,None,2))
    rnn = tf.keras.layers.RNN(cell)
    nsteps = 10;
    nbatch = 100;
    rnn((np.random.randn(nbatch,nsteps,1).astype("float32"),np.random.randn(nbatch,nsteps,1000,2).astype("float32")))
