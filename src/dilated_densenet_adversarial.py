"""
A DenseNet with dilated 1D convolutions with domain adversarial regularization.
Domain adversarial regularization is achieved by the simple gradient reversal technique.
"""





import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as pplot
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Flatten, BatchNormalization, Dropout
from tensorflow_probability import layers as tfl


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        print("custom grad")
        return tf.negative(dy)
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GradReverse,self).__init__(**kwargs)

    def call(self, x):
        return grad_reverse(x)

def make_densenet_parametrization_dictionary(n_depth = 6, filter_size = 5, kernel_size = 5, dilation_rate = 8):
    densenet_props = {"filters" : [filter_size] * n_depth,
                      "dilation_rates" : [dilation_rate] * n_depth ,
                      "kernel_sizes" : [kernel_size]*n_depth}
    return densenet_props


def make_densenet_block(xin,densenet_props, layer_prefix = "_"):
    """
    With this implementation, number of output filters per layer should remain constant.
    In order to use the same convolutional sub-network the number of input filters should
    be the same as the last time it was used. Otherwise there must be a
    transition with 1x1 convolution and the correct number of filters to the upcoming layer for
    each subsequent use of the layer. This choice 
    seems to make the network a bit harder to train.

    Batch normalization was never found to help.

    input:
      xin            : an input tensor (time-series data)
      densenet_props : a dictionary containing the parameters. For Instance:
                       densenet_props = {
                         "dilation_rates" : [8,8,8,8,8],
                         "kernel_sizes" : [kernel]*4,
                         "filters" : [nfilters]*4,
                         "Variational" : False,
                         "internal_bottleneck_factor" : 4}

                         "Variational":
                             denotes a tfp layer with reparametrization for the last layer of the densenet instead of a simple convolution.

                         "internal_bottleneck_factor":
                             controls the number of feature maps output by the feature-wise multiplication performed
                             at each composite function. See "Composite function" paragraph in paper. Default is 4 (as in Huan et al).
                             If the number of internal bottleneckk filters is not divisible with this factor, the number of bottleneck filters
                             is the closest integer.
    """
    padding_type = "same"
    xin_c1 = Conv1D( filters = densenet_props["filters"][0], kernel_size = 1, padding = padding_type, name = "DN_{}_C1D".format(layer_prefix))(xin)
    curr_state = [xin_c1];
    idx = 0

    # Some input parameter defaults management:
    if "Variational" not in densenet_props.keys():
        densenet_props["Variational"] = False

    if "internal_bottleneck_factor" not in densenet_props.keys():
        densenet_props["internal_bottleneck_factor"] = 1;

    if "dropout_after_composite" not in densenet_props.keys():
        densenet_props['dropout_after_composite'] = None

    if densenet_props['Variational']:
        from tensorflow_probability import layers
        C1DFcn = layers.Convolution1DReparameterization
    else:
        C1DFcn = Conv1D

    for dilation_rate , kernel_size, filters  in zip(densenet_props["dilation_rates"], densenet_props["kernel_sizes"],densenet_props["filters"]):

        n_bottleneck_filters = int(filters / densenet_props["internal_bottleneck_factor"])

        # Batch normalization was found to harm the preformance of the network! (tried both sample-wise and filter-wise)
        seq_layers = [C1DFcn(dilation_rate = 1, kernel_size = 1, filters = n_bottleneck_filters, padding = padding_type),
                      keras.layers.Lambda(lambda x : tf.keras.activations.relu(x)),
                      Conv1D(dilation_rate = dilation_rate, kernel_size = kernel_size, filters = filters, padding = padding_type)]

        if densenet_props["dropout_after_composite"] is not None:
            seq_layers.append(Dropout(rate = densenet_props["dropout_after_composite"]))

        layer = keras.Sequential(seq_layers,name = "DN_{}_idx_{}".format(layer_prefix,idx))


        append_state = [layer(layer_output_i) for layer_output_i in curr_state]
        curr_state = [*append_state, *curr_state]
        idx += 1

    #ipt_shape = curr_state[0].shape[1:].as_list()
    #opt_shape = curr_state[-1].shape[1:].as_list()

    #print("reduction in length:{}".format( 1 - ipt_shape[0]/opt_shape[0]))
    outname = "DN_{}_ConcatOut".format(layer_prefix);
    output = tf.keras.layers.concatenate(inputs = append_state, name = outname) if len(append_state) > 1 else keras.layers.Lambda(lambda x : x, name = outname)(append_state[0])
    print(output)
    #print(output.shape)
    #output = keras.layers.LayerNormalization(axis = -2)(output)
    #output = keras.layers.LayerNormalization(axis = -1)(output)
    return keras.layers.Lambda(lambda x : x, name = "DN_{}_Out".format(layer_prefix))(output)




def make_model(nw_params):
    """
    Creates a model with domain adversarial normalization

    The domain adversarial regularization is implemented by reversing the gradients from a discriminator
    that identifies the "domain" that we want our model to be invariant to.
    """

    ## Helper (factory) function
    def make_densenet_block_from_dict(h_curr, param_dict):
        params = param_dict["params"];
        name_pref = param_dict["id"];
        h_curr = make_densenet_block(h_curr, params, layer_prefix=name_pref)
        return h_curr

    def make_transition_from_dict(h_curr, param_dict):
        params = param_dict["params"]

        n_filters_prev = h_curr.shape.as_list()[-1]
        n_output_feature_maps = int(params["theta"] * n_filters_prev)
        h_curr = keras.layers.Conv1D(kernel_size=1, filters=n_output_feature_maps)(h_curr)
        h_curr = keras.layers.AveragePooling1D(pool_size=params["pool_size"])(h_curr)
        return h_curr

    def make_final_pooling_from_dict(hcurr, param_dict):
        return GlobalAveragePooling1D()(h_curr)

    def make_block_from_dict(h_curr, param_dict):
        """
        Write any new types of blocks here.
        This allows all of the network 
        to be parametrized by a simple JSON.
        """
        string_to_function = {
            "densenet_block" : make_densenet_block_from_dict,
            "transition_block" : make_transition_from_dict,
            "final_pooling" : make_final_pooling_from_dict
        }
        return string_to_function[param_dict["type"]](h_curr,param_dict)

    def make_classifier_from_dict(h_features, param_dicts):
        string_to_layer = {
            "Dense" : Dense,
            "Dropout" : Dropout,
            "DenseLocalReparameterization" : tfl.DenseLocalReparameterization
        }


        h_curr = h_features
        for pd_ in param_dicts:

            h_curr = string_to_layer[pd_["type"]](**pd_["params"])(h_curr)
        return h_curr
        ### Create inputs:
    print(nw_params)
    input_params = nw_params["inputs"]
    inputs = [];
    for ipt in input_params:
        in_curr = tf.keras.layers.Input(shape = ipt["shape"], name = ipt["id"])

        if ipt["id"] == "timeseries_input":
            x_in_densenet = in_curr

        if ipt["id"] == "domain_input":
            x_in_domain = in_curr
            # The timeseries input is the "data" we need to train a model on.

        inputs.append(in_curr)


    ## Create the DenseNet layer stacks:
    densenet_params = nw_params["densenet_1d"]["network_layers"]
    n_blocks = len(densenet_params)
    h_curr = x_in_densenet
    for dn_block_params in densenet_params:
        h_curr = make_block_from_dict(h_curr, dn_block_params)

    ## Add a gradient reversal layer in-between the features and the domain classifier:
    h_curr_gd = GradReverse()(h_curr)
    domain_classifier_out = make_classifier_from_dict(h_curr_gd,nw_params["domain_classifier"]["network_layers"])

    classifier_out = make_classifier_from_dict(h_curr,nw_params["classifier"]["network_layers"])

    model = Model(inputs  = inputs, outputs = [domain_classifier_out, classifier_out])

    return model


