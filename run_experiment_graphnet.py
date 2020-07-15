from collections import OrderedDict
import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)

import numpy as np
import matplotlib.pyplot as pplot
tfd = tfp.distributions
from ipywidgets import FloatSlider, IntSlider, interact, interactive


import pandas as pd

from fictitious_example.dataset_utils import *
from fictitious_example.minigraphnets import *
from datasets.femto_bearing import FEMTOBearingsDataset

def train(gtot_, dataset):
    None


def evaluate(gtot_, dataset):
    None




if __name__ == "__main__":
    femto_dataset = FEMTOBearingsDataset()
    inds_exp_target, inds_exp_source = [femto_dataset.inds_exp_target, femto_dataset.inds_exp_source]

    # This encapsulates a composition of a graph-indipendent graphnet that has a convolutional head on the input and a simple FFNN for 
    # edge features (in the RUL application this is time elapsed between observations) and a "core" network that may be applied recursively 
    # several times during evaluation (typical trick of GraphNets to propagate information without weights blowing up).
    gtot = GraphNetFunctionFactory(NETWORK_SIZE_GLOBAL = 50, USE_PRENETWORKS = True, EDGE_NODE_STATE_SIZE=15)




