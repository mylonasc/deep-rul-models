import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as pplot
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
import sys



if __name__ == "__main__":
    ndn = 4
    ndn2 = 4
    drate1 = int(sys.argv[1])
    kernsize1 = int(sys.argv[2])
    nfilters1 = int(sys.argv[3])
    drate2 = int(sys.argv[4])
    kernsize2 = int(sys.argv[5])
    nfilters2 = int(sys.argv[6])


    experiment_parametrization = {
        'dilation_rates_dn1': [drate1]*ndn,
        'kernel_sizes_dn1' : [kernsize1]*ndn,
        'nfilters_dn1' :  [nfilters1]*ndn,
        'dilation_rates_dn2' : [drate2]*ndn2,
        'kernel_sizes_dn2' : [kernsize2]*ndn2,
        'nfilters_dn2': [nfilters2]*ndn2
    }

    make_network_description_experiments_A(**experiment_parametrization)
    model_json = make_network_json_experiments_A(**experiment_parametrization)
    model = make_model(model_json)
    
    print("loading data")
    data = load_hilti_fatigue_data()

    [X,Y,Yoh, eid_vector] = data["training_instances"]
    [Xstrong,Ystrong,YstrongOH, eid_vector_strong] = data["validation_instances"]

    from sklearn.model_selection import train_test_split

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

    epochs = 100; #training_parameters["nepochs"]
    batchsize = 100;
    
    from tensorflow.math import confusion_matrix
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    
    es = keras.callbacks.EarlyStopping(monitor='val_Y_output_loss', mode='min', verbose=1, patience=10, restore_best_weights = True)
    
    def top3_acc(labels, logits):
        return keras.metrics.top_k_categorical_accuracy(labels,logits, k=3)
    
    def plot_confusion():
        cmat = confusion_matrix(np.argmax(Yoh,1),np.argmax(Yhat,1))
        pplot.pcolor(cmat.eval())
        pplot.show()
    
    adv_loss = {"ExpID" : tf.compat.v1.losses.softmax_cross_entropy }
    loss_fcn = {"Y_output" :tf.compat.v1.losses.softmax_cross_entropy,
                "ExpID" : lambda y,yhat : tf.compat.v1.losses.softmax_cross_entropy(y,yhat)}
    
    loss_w = {"Y_output": 0.66*3./(3.+15.),"ExpID" : 0.33*15./(3.+15.)}
    
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
                         callbacks=[es])
    
    ## Saving the results:
    import os
    import json
    
    desc_str = make_network_description_experiments_A(**experiment_parametrization)
    results_dir_name = os.path.join("Experiments_A",desc_str);
    try:
        results_dir = os.mkdir(results_dir_name)
    except:
        None
    save_path_losses = os.path.join(results_dir_name,"losses");
    save_path_json = os.path.join(results_dir_name,"json");
    
    import pandas as pd
    pd.DataFrame(history_d.history).to_csv(save_path_losses)
    
    with open(save_path_json,'w') as f:
        f.write(json.dumps(experiment_parametrization))

