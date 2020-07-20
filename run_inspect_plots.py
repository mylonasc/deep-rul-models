import pandas as pd
import tensorflow as tf


from collections import OrderedDict
import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Sequential,Model
import tensorflow.keras as keras

import socket
if socket.gethostname() == 'marvin':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

import numpy as np
import matplotlib.pyplot as pplot
tfd = tfp.distributions
from ipywidgets import FloatSlider, IntSlider, interact, interactive
from datasets.femto_bearing import FEMTOBearingsDataset
from minigraphnets import Edge,Node, Graph
from graphnet_utils import GraphNetFunctionFactory, GraphNet


import sys

if __name__ == "__main__":
    ## Load model:
    model_path = sys.argv[1]
    pd.read_pickle("models/runs_dataframe")
    p = pd.read_pickle("models/runs_dataframe")
    p['min_val_loss'] = p['losses'].apply(lambda x : np.min(x['val_loss']))
    
    model_row = p[p['model_path'] == model_path].iloc[0]

    gn_tot = GraphNetFunctionFactory.make_from_record(model_row)
    gn_tot.load(model_path)

    self = gn_tot

    def eval_graphnets(graph_data_, iterations = 5, eval_mode = "batched", return_reparametrization = True):
        """
        graph_data_  : is a "graph" object that contains a batch of graphs (more correctly, a graph tuple as DM calls it)
        iterations   : number of core iterations for the computation.
        return_distr_params : return the distribution parameters instead of the distribution itself. This is in place because of some buggy model loading (loaded models don't return distribution objects).
        """
        graph_out = self.graph_indep.graph_eval(graph_data_,eval_mode = eval_mode)
        for iterations in range(iterations):
            graph_out = self.core.graph_eval(graph_out, eval_mode = eval_mode) + graph_out # Addition adds all representations (look at implementation of "Graph")

        # Finally the node_to_prob returns a reparametrized "Gamma" distribution from only the final node state
        if not return_reparametrization:
            return self.core.node_to_prob_function(graph_out.nodes[-1].node_attr_tensor)
        else:
            v = self.core.node_to_prob_function.get_layer("output")(graph_out.nodes[-1].node_attr_tensor)
            return _instantiate_gamma(v)

    def _instantiate_gamma(t, NParams_ = 1):
        return tfd.Gamma(concentration = t[...,0:NParams_], rate = t[...,NParams_:2*NParams_])


    femto_dataset = FEMTOBearingsDataset()


    from utils import get_graph_data
    experiments_to_plot = [1]
    def plot_experiments(experiments_to_plot):
    #if True:
        
        #training = inds_exp_source
        nsampled = 500

        #pplot.figure(figsize = (15,10), dpi = 150)
        
        pplot.figure(figsize = (15,10), dpi = 75)
        
        nnodes_list = [1,2,15]
        nseq_len = [100,100,200]
        minspacing= [10,10,10]
        gnsteps  = [ 5,5,5]
        

        normalization_factor_time = femto_dataset.normalization_factor_time

        kk = 0;
        for ee in experiments_to_plot:
            for nnodes, gnsteps_,nseq_,minspacing_ in zip(nnodes_list, gnsteps, nseq_len, minspacing):
                #ee = training[0]
                graphs, y_times = get_graph_data(ee, X_ = femto_dataset.X, eid_oh_ = femto_dataset.eid_oh,
                                                 yrem_norm_ = femto_dataset.yrem_norm, n_sampled_graphs = nsampled, 
                                                 nnodes=nnodes, fixed_spacing_indices=False, min_spacing=minspacing_,
                                                 nseq_range=nseq_)
                probs = eval_graphnets(graphs,gnsteps_, eval_mode="safe")
                #eval_graphnets()
                ids_sorted = np.argsort(y_times)
                time_grid = np.linspace(np.min(y_times),np.max(y_times), 150);
                time_grid = np.linspace(np.min(y_times), 60000./normalization_factor_time, 150)
                #time_grid = np.linspace(np.min(y_times),3.5, 150);
                
                e_y = probs.mean()
                p_y = probs.prob(time_grid).numpy().T

                y_times_sorted = y_times[ids_sorted];
                pplot.subplot( len(experiments_to_plot),len(nnodes_list), kk+1)
                pplot.pcolor([r for r in range(p_y.shape[1])], time_grid*normalization_factor_time, p_y[:,ids_sorted]**0.5, cmap = "gray")
                pplot.plot(y_times_sorted  *normalization_factor_time)
                q90 = np.quantile(probs.sample(1000).numpy(),[0.05,0.95],0)[:,:,0].T[ids_sorted]

                #pplot.plot(e_y.numpy()[ids_sorted]*normalization_factor_time,'C1',label = "$E[t_f]$",alpha = 0.5)<3
                pplot.plot(q90*femto_dataset.normalization_factor_time,'C1-.', alpha = 0.4)
                pplot.ylim(0,75000)
                pplot.xlabel("Sample")
                pplot.ylabel("Time to failure [s]")

                nll = -np.mean(probs.log_prob(y_times[np.newaxis].T))

                title = "Experiment %i\n Observations:%i \nnll:%2.3f"%(ee,nnodes,nll)
                pplot.title(title)
                kk+=1
                #p_y.shape
                #pplot.show()

        pplot.subplots_adjust(hspace = 0.75, wspace = 0.7)
        
    #unseen =  inds_exp_source[3:6]#[4:7]#inds_exp_source[0:3] #inds_exp_target[0:3]

    model_hash = model_path[7:-9] 

    #model_row
    plot_experiments(femto_dataset.inds_exp_target)

    pplot.savefig(os.path.join("models/figures" , str(model_hash) + "_target.png")) ; 
    plot_experiments(femto_dataset.inds_exp_source[0:3])
    pplot.savefig(os.path.join("models/figures" , str(model_hash) + "_source1.png")) ; 
    plot_experiments(femto_dataset.inds_exp_source[3:6])
    pplot.savefig(os.path.join("models/figures" , str(model_hash) + "_source2.png")) ;
