import pandas as pd
import sys
import os

if __name__ == "__main__":
    h = sys.argv[1]
    f = os.path.join('models','runs_dataframe')
    p = pd.read_pickle(f)
    
   

    res = p[[h in v for v in p.model_path.values]]
    assert(len(res) == 1)
    columns = res.columns
    res = res.iloc[0]


    model_options = {'network_size_global' : int(res.network_size_global),
            'edge_node_state_size' : int(res.edge_node_state_size),
            'use_prenetworks' : bool(res.use_prenetworks),
            'graph_function_output_activation': res.graph_function_output_activation}

    

    val_defaults_dict = {'n_conv_blocks' : 3 , 
            "nfilts2" : 50,
            'nfilts' : 18,
            'ksize' : 3,
            'conv_block_activation_type' : 'leaky_relu'
            }
    def check_set_default(val):
        if val not in columns:
            res_ = val_defaults_dict[val]
            return res_

    
    for k in val_defaults_dict.keys():
        model_options.update({k : check_set_default(k)})


    # structure to sets of consistent inputs to the training function
    #res_input = " ".join([p_ for p_ in params])
    #print(res_input)
    import json
    json_written_at =  "%s_model.json"%h
    with open(json_written_at, 'w') as f:
        f.write(json.dumps(model_options,indent = 4))
    print(json_written_at)






            

