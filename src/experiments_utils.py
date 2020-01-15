

def make_network_description_experiments_A(dilation_rates_dn1, kernel_sizes_dn1, nfilters_dn1, dilation_rates_dn2, kernel_sizes_dn2,nfilters_dn2):
    get_dn_desc = lambda ff,kk,dd : "(F"+".".join([str(cc) for cc in ff])+"_K"+".".join([str(cc) for cc in kk])+"_D"+".".join([str(cc) for cc in dd]) + ")"
    dn1_desc = get_dn_desc(nfilters_dn1, kernel_sizes_dn1, dilation_rates_dn1)
    dn2_desc = get_dn_desc(nfilters_dn2, kernel_sizes_dn2, dilation_rates_dn2)

    experiment_name_desc = "Training_ExpA_ArchDN1"+dn1_desc+"_ArchDN2"+dn2_desc
    return experiment_name_desc

def make_network_json_experiments_A(dilation_rates_dn1, kernel_sizes_dn1, nfilters_dn1, dilation_rates_dn2, kernel_sizes_dn2,nfilters_dn2, 
                                    input_shape = (1500,6), ndomain_classes = 3, nprediction_classes = 15):
    """
    Investigate the effect of dilation rates, kernel sizes and number of layers per DenseNet block. 
    Using 2 DenseNet blocks.
    """    
    network_parametrization = {
        "inputs" : 
        [
            {
                "type" : "timeseries",
                "help_string" : "(required) the timeseries input for the layer.",
                "id" : "timeseries_input",
                "shape" : input_shape,
            },
        ]
    }
    
    network_layers = [
            {
                "type" : "densenet_block",
                "id" : "DN1",
                "help_string" : "Parametrization for a `densenet_block`",
                "params" : {
                    "filters" : nfilters_dn1,
                    "dilation_rates" : dilation_rates_dn1,
                    "kernel_sizes" : kernel_sizes_dn1
                }
            },
            {
                "type" : "transition_block",
                "id" : "T1",
                "help_string" : "theta: the shrinking factor for the number of feature maps \n pool_size: the pooling size for the transition layer (AveragePooling1D)",
                "params" : {
                    "pool_size" : 3,
                    "theta"  : 0.5
                }
            },
            {
                "type" : "densenet_block",
                "id" : "DN2",
                "help_string" : "Parametrization for a `densenet_block`",
                "params" : {
                    "filters" : nfilters_dn2,
                    "dilation_rates" : dilation_rates_dn2,
                    "kernel_sizes" : kernel_sizes_dn2
                }
            },
            {
                "type" : "final_pooling",
                "id" : "FinalPooling"            
            }
    ]
    network_parametrization.update({"densenet_1d" : {"network_layers" : network_layers}})
    network_parametrization.update({
        "domain_classifier" : {            
            "network_layers" : 
            [
                {
                    "type" : "Dense",
                    "params" : {"units" : 100,"activation" : "relu", "use_bias" : True}
                },
                {
                    "type" : "Dropout",
                    "params" : {"rate"  : 0.2}
                },
                {
                    "type" : "Dense",
                    "id" : "domain_output",
                    "params" : {"units" : 3, "name" : 'ExpID'}
                }
            ]
        },
        "classifier" : {
            "network_layers" : 
            [
                {
                    "type" : "Dense",
                    "params" : {"units" : 100,"activation" : "relu", "use_bias" : True}
                },
                {
                    "type" : "Dropout",
                    "params" : {"rate"  : 0.2}
                },
                {
                    "type" : "Dense",
                    "id" : "domain_output",
                    "params" : {"units" : 15, "name" : "Y_output"}
                }
            ]
        }
        
    })
    return network_parametrization
