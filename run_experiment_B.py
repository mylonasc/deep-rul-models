######################################################
# Selection of relative weights between adv.loss and #
# prediction loss                                    #
######################################################
import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow.keras as keras
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Flatten, BatchNormalization, Dropout

## In my GPU this is needed - no idea why. Otherwise I get "failed to get convolution algorithm"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)
tf.keras.backend.set_session(sess)

from src.dilated_densenet_adversarial import make_model
from src.dilated_densenet_adversarial import make_densenet_parametrization_dictionary

from src.experiments_utils import *
from src.dilated_densenet_adversarial import *

from src.util import load_hilti_fatigue_data
from src.util import plot_confusion_matrix

import sys

from sklearn.model_selection import train_test_split


def transform_model_json_to_ExpB(model_json, nclasses = 10):
    """
    Removes the intermediate "transition" layer and the final layer I was using in the initial experiments.
    Turns out that there was an error in the implementation that made most of the network innactive and train fast, 
    where at the same time I was getting good validation accuracy. Changed to an actual denseNet (with some modifications)
    to have richer features and more larger rec. field (indeed works better).
    """

    model_json["classifier"]["network_layers"][-1]['params']['units'] = nclasses
    model_json["densenet_1d"]["network_layers"][0]["params"]["Variational"] = False
    model_json["densenet_1d"]["network_layers"] = [model_json["densenet_1d"]["network_layers"][0],model_json["densenet_1d"]["network_layers"][-1]]
    return model_json

def transform_model_json_to_ExpC(model_json, nclasses = 10):
    """
    Removes the intermediate "transition" layer and the final layer I was using in the initial experiments.
    Turns out that there was an error in the implementation that made most of the network innactive and train fast, 
    where at the same time I was getting good validation accuracy. Changed to an actual denseNet (with some modifications)
    to have richer features and more larger rec. field (indeed works better).

    Also uses a Local Reparametrized variational layer for the (non-adversarial) predictor.
    """

    model_json = transform_model_json_to_ExpB(model_json, nclasses = nclasses)


    return model_json



if __name__ == "__main__":

    c_loss = float(sys.argv[1])

    ndn = 4
    ndn2 = 4

    experiment_parametrization = {
        'dilation_rates_dn1': [4,8,16,32,4],
        'kernel_sizes_dn1' : [5,5,5,5,5,5],
        'nfilters_dn1' :  [20]*ndn,
        'dilation_rates_dn2' : [4]*ndn2, #unused
        'kernel_sizes_dn2' : [5]*ndn2, #unused
        'nfilters_dn2': [10]*ndn2 #unused
    }

    
    print("loading data")
    #data = load_hilti_fatigue_data(leave_exp_out='None')
    #data = load_hilti_fatigue_data(keep_from_end = 100000,leave_exp_out="None",stage_2_from_disk=True, nclasses = 10)
    data = np.load("experiments_b_data.npy", allow_pickle = True)
    data = data[()]

    # Train while leaving a whole experiment out, report accuracy in held-out experiment:
    for leave_exp_out in ["VA_1", "VA_2", "VA_3" , "VA_4"]:

        import os
        import json
        
        desc_str = make_network_description_experiments_A(**experiment_parametrization)
        results_dir_name = os.path.join("Experiments_B",desc_str);
        try:
            results_dir = os.mkdir(results_dir_name)
        except:
            None
        save_path_losses = os.path.join(results_dir_name,"losses");
        save_path_json = os.path.join(results_dir_name,"json");
        save_path_figures = results_dir_name 

        print("=="*10)
        print("Training without experiment: %s"%leave_exp_out)

        make_network_description_experiments_A(**experiment_parametrization)
        model_json = make_network_json_experiments_A(**experiment_parametrization)
        model_json = transform_model_json_to_ExpC(model_json)
        model = make_model(model_json)

        [X_all,Y_all,Yoh_all, eid_vector_all] = [cc for cc in data["training_instances"]]
        [X,Y,Yoh, eid_vector] = [cc[eid_vector_all != leave_exp_out] for cc in [X_all,Y_all,Yoh_all, eid_vector_all]]
        [Xstrong,Ystrong,YstrongOH, eid_vector_strong] = [cc[eid_vector_all== leave_exp_out] for cc in [X_all,Y_all,Yoh_all, eid_vector_all]]


        Y_classes = np.argmax(Yoh,1)
        EidOH = np.zeros([eid_vector.shape[0],len(np.unique(eid_vector))])
        nexperiments = len(np.unique(eid_vector));
        for i in np.unique(eid_vector):
            EidOH[eid_vector == i,np.where(np.unique(eid_vector) == i)[0]] = 1

        Y_OH = np.zeros([Y.shape[0],len(np.unique(Y_classes))])
        for i in np.unique(Y_classes):
            Y_OH[Y_classes == i,np.where(np.unique(Y_classes) == i)[0]] = 1

        Xtrain, Xtest,Ytrain, Ytest, EIDTrain, EIDTest = train_test_split(X, Y_OH, EidOH,
                                                                          stratify = Y_classes,
                                                                          train_size = 0.80, random_state = 168)

        ## Actual training:

        epochs = 100 ; #training_parameters["nepochs"]
        batchsize = 100;
        
        from tensorflow.math import confusion_matrix
        import tensorflow.keras as keras
        from tensorflow.keras import layers
        
        es = keras.callbacks.EarlyStopping(monitor='val_Y_output_loss', mode='min', verbose=1, patience=10, restore_best_weights = True)

        top3acc = []
        
        def top3_acc(labels, logits):
            return keras.metrics.top_k_categorical_accuracy(labels,logits, k=3)
        
        def plot_confusion():
            cmat = confusion_matrix(np.argmax(Yoh,1),np.argmax(Yhat,1))
            pplot.pcolor(cmat.eval())
            pplot.show()

        def plot_cmatrix_strong_return_top3acc(leave_exp_out, epoch, c_loss,root_folder = results_dir_name):
            print("in plot cmatrix")
            yyhat = model.predict(Xstrong)
            print([(i, s.shape) for i,s in enumerate(yyhat)])
            yyhat = yyhat[1]
            cmat = confusion_matrix(np.argmax(YstrongOH,1),np.argmax(yyhat,1))
            #cmat = cmat[0:-1,0:-1]
            crange = int(150000/15);
            target_names = ["%i>=Nr>%i"%((i)*crange,(i+1)*crange) for i in range(cmat.shape[0])]
            if (epoch+1) % 10 == 0:
                plot_confusion_matrix(cmat.eval(session = keras.backend.get_session()),
                                      target_names = target_names , figsize = (10,10), normalize= True, title = "Confusion Matrix\nLeft out exp%s"%(leave_exp_out))

                pplot.savefig(os.path.join(root_folder,"epoch_%03i_exp_%s_closs%f.png"%(epoch,leave_exp_out, c_loss)))

            
            print("exiting plot cmatrix")

            return keras.metrics.top_k_categorical_accuracy(YstrongOH,yyhat, k=3).eval(session = keras.backend.get_session())

        top3_strong_list = []
        class Top3Strong(tf.keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs = None):
                print("on epoch end")
                top3_strong = plot_cmatrix_strong_return_top3acc(leave_exp_out, epoch, c_loss,root_folder = save_path_figures)
                top3_strong = np.mean(top3_strong)
                top3_strong_list.append(top3_strong)

                pplot.close()

        top3strong_eval = Top3Strong()


        
        adv_loss = {"ExpID" : tf.compat.v1.losses.softmax_cross_entropy }
        loss_fcn = {"Y_output" :tf.compat.v1.losses.softmax_cross_entropy,
                    "ExpID" : lambda y,yhat : tf.compat.v1.losses.softmax_cross_entropy(y,yhat)}
        
        loss_w = {"Y_output": 0.66*3./(3.+15.)*(1.-c_loss),"ExpID" : 0.33*15./(3.+15.)*c_loss}
        
        # GOAL: Minimize the useless discriminator while maximizing the useful classifier.
        
        # useful discriminator:
        model_opt = keras.optimizers.Adam(learning_rate=0.001);
        model.compile(optimizer =  model_opt,
                      loss =loss_fcn,
                      loss_weights = loss_w,
                      metrics = {"Y_output" : top3_acc})
        
        train_in, train_out = [{"timeseries_input": Xtrain},{"Y_output":Ytrain,  "ExpID" : EIDTrain}]
        test_in, test_out = [{"timeseries_input" : Xtest}, {"Y_output" : Ytest, "ExpID" : EIDTest}]
        
        history_d = model.fit(train_in, train_out, epochs=epochs,
                             batch_size=batchsize,
                              validation_data = [test_in, test_out],# "Rc_output" :  RemCTest}],
                             callbacks=[es, top3strong_eval])
        
        ## Saving the results:

        
        
        import pandas as pd
        ## Write to file or append to file if file exists.
        ## Append results for all left-out experiments
        dict_to_df = history_d.history
        dict_to_df['LeftOut'] = leave_exp_out
        dict_to_df['c_loss'] = c_loss
        dict_to_df['top3_strong'] = top3_strong_list
        new_df = pd.DataFrame(dict_to_df)

        with open(save_path_losses, 'a') as f:
            new_df.to_csv(f, header=f.tell()==0)
        
        with open(save_path_json,'w') as f:
            f.write(json.dumps(experiment_parametrization))

