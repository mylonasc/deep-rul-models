from collections import OrderedDict
import sys
import time
import datetime

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

from datasets.femto_bearing import FEMTOBearingsDataset
from minigraphnets import Graph, Node, Edge
from graphnet_utils import GraphNetFunctionFactory, GraphNet

from utils_train import EarlyStopping, LossLogger, LRScheduler

from tqdm import tqdm
import time

from utils import get_multi_batch

def train(gn_tot, dataset,training_options):
    learning_rate = training_options['learning_rate'];
    epochs        = training_options['epochs']
    nbatch        = training_options['batch']

    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_log = LossLogger()
    early_stop = EarlyStopping(20,loss_log.loss_history['val_loss'])
    lr_scheduler = LRScheduler(opt, base_lr = learning_rate,
                               epoch_decay = 50,
                               decay_rate = 0.99,
                               burnin_epochs = 10)

    def get_multi_batch_femto(*args, **kwargs):
        new_args = (args[0], dataset)
        return get_multi_batch(*new_args, **kwargs)

    val_pct = 0.1
    fixed_spacing_indices = False;
    nnodes_schedule =     training_options['schedule_nnodes']#[1,2,5] #[10,10,1,2,5]#2,2,3,3]#,1,2,2,1,1,3,3]
    nseq_range_schedule = training_options['nseq_length']
    min_spacing_schedule = training_options['schedule_min_sep'];
    iterations_schedule = training_options['iterations_schedule']
    RAND_SEED = training_options['rand_seed']

    for i in range(0,epochs):
        np.random.seed(RAND_SEED);
        epoch = i

        # This makes sure that I always get different training and validation sets (there can still be some overlap but training should work ok)
        nnodes      = nnodes_schedule[i%len(nnodes_schedule)]
        nseq_range  = nseq_range_schedule[i%len(nseq_range_schedule)]
        min_spacing = min_spacing_schedule[i%len(min_spacing_schedule)]
        iterations = iterations_schedule[i%len(iterations_schedule)]

        if (nnodes * min_spacing) > nseq_range:
            min_spacing = 10
            nseq_range = nnodes * min_spacing+10

        print("nnodes: %i, seq_range %i epoch: %i"%(nnodes, nseq_range, i))
        curr_data = get_multi_batch_femto(nbatch, source_ds=True,nnodes=nnodes,
                                    min_spacing = min_spacing,
                                    nseq_range = nseq_range,
                                    fixed_spacing_indices=fixed_spacing_indices);

        loss_epoch = 0;
        val_loss_epoch = 0;

        for single_minibatch in tqdm(curr_data):
            with tf.GradientTape() as tape:
                graph_curr, y_curr = single_minibatch;
                #print(graph_curr)
                #def eval_graphnets_loss(graph_curr_, ycurr_,iterations):
                prob_out = gn_tot.eval_graphnets(graph_curr.copy(), iterations)
                loss_vals = -prob_out.log_prob(y_curr[np.newaxis].T)
                #ycurr_t = y_curr[np.newaxis].T
                #ycurr_t = tf.Variable(ycurr_t)i
                #loss_vals=  eval_graphnets_loss(graph_curr.copy(), ycurr_t, tf.constant(iterations))

                all_weights =gn_tot.weights()


                train_loss = loss_vals[0:int(nbatch*(1-val_pct))];

                grads = tape.gradient(train_loss, all_weights)
                all_weights_filt = [all_weights[k] for k in range(len(grads)) if grads[k] is not None]
                grads_filt = [grads[k] for k in range(len(grads)) if grads[k] is not None]
                opt.apply_gradients(zip(grads_filt, all_weights_filt))

                loss_epoch += train_loss/len(y_curr)

            val_loss_epoch += tf.reduce_mean(loss_vals[int(-nbatch*(val_pct)):])

        loss_log.append_loss(np.sum(loss_epoch.numpy()))
        loss_log.append_val_loss(np.sum(val_loss_epoch.numpy()))
        loss_log.print()
        lr_scheduler.on_epoch_end(epoch)
        if (early_stop.on_epoch_end(epoch) )and (epoch  > 20):
            break

        #if (epoch)%10 == 0:
        #    pplot.plot(loss_log.loss_history['loss'])
        #    pplot.plot(loss_log.loss_history['val_loss'])
    


    return loss_log.loss_history




if __name__ == "__main__":
    rundf_path = os.path.join("models_sept20_runs","runs_dataframe")
    femto_dataset = FEMTOBearingsDataset()
    inds_exp_target, inds_exp_source = [femto_dataset.inds_exp_target, femto_dataset.inds_exp_source]

    if sys.argv[1] == '--from-model-json':
        json_file = sys.argv[2]
        import json
        with open(json_file,'r') as f:
            model_options = json.loads(f.read())
            f.close()
    else:

        network_size_global = int(sys.argv[1])
        graphstate_size = int(sys.argv[2])
        gn_fn_output_activation = sys.argv[3]

        if len(sys.argv)>4:
            n_conv_blocks = sys.argv[4]
            nfilts = sys.argv[5]
            nfilts2 = sys.argv[6]
            ksize = sys.argv[7]
            conv_block_activation_type = 'leaky_relu'
        else:
            n_conv_blocks = 3
            nfilts2 = 50
            nfilts = 18
            ksize = 3
            conv_block_activation_type = 'leaky_relu'

        experiment_metadata = {"hpr_id" : "1", "description" : "larger graph-states seem to help. The best runs seemed to be with width parameter only 15! Investigating now the effect of CNN parameters."}

        model_options = {'network_size_global' : network_size_global, 
                         'edge_node_state_size' : graphstate_size, 
                         'use_prenetworks' : True,
                         'graph_function_output_activation' : gn_fn_output_activation}

        model_options.update( {'n_conv_blocks' : int(n_conv_blocks) ,
                    'nfilts' : int(nfilts), 
                    'nfilts2' : int(nfilts2), 
                    'ksize': int(ksize) ,
                    'conv_block_activation_type' : conv_block_activation_type})
        print(model_options)



    dataset_options = femto_dataset.get_dataset_config()

    training_options = {'learning_rate' : 0.001,
                        'schedule_nnodes' :  [1,2,5,10,15,20],
                        'schedule_min_sep': [10],
                        'nseq_length' : [300],
                        'iterations_schedule' : [5],
                        'epochs':300,
                        'batch' : 150,
                        'rand_seed' : 42}
    if sys.argv[3] == '--training-options-json':
        training_options_json = sys.argv[4]

#    best_training_options = {'learning_rate' : 0.001,
#                        'schedule_nnodes' :  [1,2,5],
#                        'schedule_min_sep': [10, 20],
#                        'nseq_length' : [100],
#                        'iterations_schedule' : [5],
#                        'epochs':300,
#                        'batch' : 300,
#                        'rand_seed' : 42}
        with open(training_options_json,'r') as f:
            training_options = json.loads(f.read())

    


    # This encapsulates a composition of a graph-indipendent graphnet that has a convolutional head on the input and a simple FFNN for 
    # edge features (in the RUL application this is time elapsed between observations) and a "core" network that may be applied recursively 
    # several times during evaluation (typical trick of GraphNets to propagate information without weights blowing up).
    gtot = GraphNetFunctionFactory( **model_options)
    femto_dataset = FEMTOBearingsDataset()
    gtot.make_graphnet_comp_blocks(femto_dataset.X[0].shape[0])
    
    # Training of the model:

    time_started= datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    losses = train(gtot, femto_dataset, training_options)


    ## Save model and training loop outputs:
    import hashlib
    import json
    
    # A model path is computed as a hash of the training parameters, the 
    training_params_hash =  hashlib.md5(json.dumps(training_options).encode("utf-8")).hexdigest()
    model_hash = gtot.get_hash().hexdigest()
    total_hash = hashlib.md5((model_hash + training_params_hash).encode('utf-8')).hexdigest()
    model_path = os.path.join("models","%s.graphnet"%total_hash)

    gtot.save(model_path)
    

    time_finished = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    values_list = [model_options, training_options, losses,model_path, dataset_options,time_finished, time_started]
    keys_list = ['model_options','training_options','losses','model_path','dataset_options','time_finished','time_started']
    dat = {}
    for k,v in zip(keys_list,values_list):
        if k in ['model_options','training_options','dataset_options']:
            for k_, v_ in zip(v.keys(), v.values()):
                dat.update({k_ : [v_]})
                continue
        else:
            dat.update({k : [v]})


    dat.update({"experiment_metadata_id" : [experiment_metadata['hpr_id']] , "experiment_metadata_desc" : experiment_metadata['description']})

    df = pd.DataFrame(dat)

    if not os.path.exists(rundf_path):
        df.to_pickle(rundf_path)
    else:
        df_prev = pd.read_pickle(rundf_path)
        dfnew = df.append(df_prev)
        dfnew.to_pickle(rundf_path)
    print("all run successfully, exiting.")





