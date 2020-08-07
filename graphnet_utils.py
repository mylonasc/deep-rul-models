import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Sequential,Model
import tensorflow.keras as keras

import inspect


def _instantiate_gamma(t, NParams_ = 1):
    return tfd.Gamma(concentration = t[...,0:NParams_], rate = t[...,NParams_:2*NParams_])

class GraphNetFunctionFactory:
    def __init__(self, network_size_global= 50, use_prenetworks= True, edge_node_state_size= 15, graph_function_output_activation = "gated_tanh", 
            n_conv_blocks = 3, nfilts = 18, nfilts2 = 50, ksize = 3, conv_block_activation_type = 'leaky_relu', channels_in = 2):
        """
        Summary: 
          A factory for graphnet functions. It is custom-made for the problems of RUL from time-series. 
          It can be adapted to other prediction models. All models (except the aggregation function) are 
          relatively small MLPs terminated by sigmoid()*tanh() activation (simple tanh could also work).
          
        network_size_global: A parameter controlling the width of different networks involved.
    
        use_prenetworks:     Use a deeper architecture (see code)
        
        edge_node_state_size:  the size of the node states, and edge states. This is needed
                               to create consistent graph functions. Eventhough here I'm using the same global size,
                               the sizes of edge states and node states can be different.

        graph_function_activation: controls how the graph functions are terminated. The special option "gated_tanh" is the default (RNN/Wavenet-like activation). Original graphnets had ReLU.

        """
        self.network_size_global =network_size_global 
        self.use_prenetworks = use_prenetworks 
        self.edge_and_node_state_size = edge_node_state_size 
        self.graph_function_output_activation = graph_function_output_activation
        self.model_constr_dict= str(inspect.getargvalues(inspect.currentframe()).locals)
        self.model_str = str(self.model_constr_dict)
        # Passed with other vargs on construction:
        self.cnn_params = {
                'n_conv_blocks' : n_conv_blocks ,
                'nfilts' : nfilts, 
                'nfilts2' : nfilts2, 
                'ksize': ksize ,
                'activation_type' : conv_block_activation_type,
                'channels_in' : channels_in
                }

    @staticmethod
    def make_from_record(record):
        """
        Method to easilly create the object from a record.
        Subsequently it is loaded from disk.
        """

        s = inspect.signature(GraphNetFunctionFactory.__init__)

        # Depending on whether the input is a dictionary or a pd dataframe, transform the 
        # keys to a list in order to pass them to the constructor.
        import pandas as pd
        record_type_to_list_transformation = {
                    pd.core.series.Series : lambda x : list(x.index),
                    dict : lambda k : [k for k in x.keys()]
                }

        l_ = record_type_to_list_transformation[type(record)](record)

        l = [s_ for s_ in s.parameters.keys() if s_ in l_]

        return GraphNetFunctionFactory(**{k_:record[k_] for k_ in l})
        

    def get_hash(self):
        import hashlib
        return hashlib.md5(self.model_str.encode("utf-8"))

    def make_gamma_node_observation_mlp(self, n_node_state_output):
        """
        Takes as input a node state and returns a gamma probability distribution
        """
        seq = keras.Sequential()
        NParams= 1;
        NSamples = 100;
        #seq.add(Dense(n_gamma_internal, use_bias = True, activation = "relu", name = "output1"))
        seq.add(Dense(NParams*2, use_bias = False, activation = lambda x : tf.nn.softplus(x),name = "output"));
        # Change that in the future to the _instantiate_gamma() version (out of class)
        def instantiate_gamma(t):
            return tfd.Gamma(concentration = t[...,0:NParams], rate = t[...,NParams:2*NParams])

        #seq.add()
        seq.add(            
            tfp.layers.DistributionLambda(
                make_distribution_fn = lambda t: instantiate_gamma(t),
                convert_to_tensor_fn= lambda s : s.sample(NSamples)))
        seq._set_inputs(tf.keras.layers.Input((n_node_state_output,)))
        return seq

    def weights(self):
        return [*self.core.weights(), *self.graph_indep.weights()];

    def make_edge_function(self,n_edge_state_input = None,n_edge_state_output = None, n_node_state_input = None):
        edge_state_in = Input(shape = (n_edge_state_input), name = "edge_state");
        node_state_sender_in = Input(shape = (n_node_state_input), name = "node_sender");
        node_state_receiver_in = Input(shape = (n_node_state_input), name = "node_receiver");

        ## Building the edge MLP:
        edge_out = keras.layers.concatenate([edge_state_in, node_state_sender_in, node_state_receiver_in])

        if self.use_prenetworks:
            edge_out = Dense(self.network_size_global,  use_bias = False,name = "edge_input")(edge_out)
            edge_out = Dropout(rate = 0.2)(edge_out)

        edge_out_gate = Dense(n_edge_state_output, activation = "sigmoid", use_bias = False,name = "edge_fcnA")(edge_out)
        edge_outB = Dense(n_edge_state_output, activation = "tanh", use_bias = False,name = "edge_fcnB")(edge_out)
        edge_out =  edge_outB * edge_out_gate #+ (1 - edge_out_gate) * edge_state_in


        edge_mlp = Model(
            inputs = [edge_state_in, node_state_sender_in, node_state_receiver_in ] ,
            outputs = edge_out)

        return edge_mlp
    
    def save(self, path):
        gi_path = os.path.join(path,"graph_independent")
        core_path = os.path.join(path,"core")
        if not os.path.exists(path):
            os.makedirs(gi_path)
            os.makedirs(core_path)
            
        self.core.save(core_path)
        self.graph_indep.save(gi_path)
    
    def load(self,path):
        self.gi_path = os.path.join(path,"graph_independent")
        self.core_path = os.path.join(path,"core")
        
        self.core        = GraphNet.make_from_path(self.core_path)
        self.graph_indep = GraphNet.make_from_path(self.gi_path)

    def make_edge_function_gi(self,n_edge_state_input = None, n_edge_state_output = None, n_node_state_input = None):
        # for graph independent.
        if n_edge_state_output is None:
            n_edge_state_output = n_edge_state_input

        edge_state_in = Input(shape = (n_edge_state_input), name = "edge_state");
        edge_out = edge_state_in

        if self.use_prenetworks:
            edge_out = Dense(self.network_size_global,  use_bias = True, name = "edge_gi_input_fcn1")(edge_out)
            edge_out = tf.keras.layers.LeakyReLU()(edge_out)
            edge_out = Dropout(rate = 0.2)(edge_out)
            edge_out = Dense(self.network_size_global,  use_bias = True, name = "edge_gi_input_fcn2")(edge_out)
            edge_out = tf.keras.layers.LeakyReLU()(edge_out)

#        if self.graph_function_output_activation == 'gated_tanh':
#            edge_out_gate = Dense(n_edge_state_output, use_bias = False, activation = "sigmoid", name = "edge_gi_fcnA")(edge_out)
#            edge_outB = Dense(n_edge_state_output, use_bias = False, activation = "tanh", name = "edge_gi_fcnB")(edge_out)
#            edge_out = edge_outB * edge_out_gate 
#        else:
#            edge_out_gate = Dense(n_edge_state_output, use_bias = False, activation = self.graph_function_output_activation, name = "edge_gi_fcnA")(edge_out)
        edge_out = self.network_function_output(edge_out, 
                name_prefix = "edge_gi",
                output_size = n_edge_state_output) # Attention! Reads parameters from the factory class. Written for avoiding code repetition, not for clarity.

        edge_mlp = Model(inputs = edge_state_in,outputs = edge_out)
        return edge_mlp

    def network_function_output(self,tensor_in,name_prefix = None, output_size = None): 
        """
        Implement the gated_tanh output head and treat it uniformly with other options for the output network options (useful for hyperparameter searches)
        """

        if self.graph_function_output_activation== 'gated_tanh': # not realy an activation...
            _out_gate = Dense(output_size, use_bias = False, activation = "sigmoid", name = "%s_fcnA"%name_prefix)(tensor_in)
            _outB = Dense(output_size, use_bias = False, activation = "tanh", name = "%s_fcnB"%name_prefix)(tensor_in)
            _out = _outB * _out_gate
            #_mlp = Model(inputs = tensor_in,outputs = _out)
        else:
            _out = Dense(output_size, use_bias = False, activation = self.graph_function_output_activation, name = "%s_fcn"%name_prefix)(tensor_in)
            #edge_mlp = Model(inputs = tensor_in,outputs = _out)

        return _out

    @classmethod
    def make_edge_aggregation_function(self,edge_out_shape):
        xin = tf.keras.layers.Input(shape = (None,edge_out_shape))
        xout = tf.reduce_mean(xin,0)
        return Model(inputs = xin, outputs= xout)


    def make_node_function(self,n_edge_state_input = None,n_node_state_input = None):
        agg_edge_state_in = Input(shape = (n_edge_state_input), name = "edge_state_agg");
        node_prev_state = Input(shape = (n_node_state_input), name = "node_sender");

        ## Building the edge MLP:
        node_out = keras.layers.concatenate([agg_edge_state_in, node_prev_state]);

        if self.use_prenetworks:
            node_out = Dense(self.network_size_global,  use_bias = True,name = "node_fcn1")(node_out)
            node_out = Dropout(rate = 0.2)(node_out)
            node_out = tf.keras.layers.LeakyReLU()(node_out)
            node_out = Dense(self.network_size_global,  use_bias = True , name = "node_fcn2")(node_out)
            node_out = tf.keras.layers.LeakyReLU()(node_out)
        #node_out = Dense(n_node_state_input, use_bias = False)(node_out)

        node_out_nl = Dense(n_node_state_input, activation = "tanh", use_bias = False,name = "node_fcn_nl")(node_out)
        node_out_gate = Dense(n_node_state_input, activation = "sigmoid", use_bias = False,name = "node_fcn_gate")(node_out)
        node_out = node_out_nl * node_out_gate# + node_prev_state * (1-node_out_gate)

        node_out_model = Model(inputs = [agg_edge_state_in, node_prev_state] ,outputs = node_out)

        return node_out_model



    def make_conv_input_head_node_function(self,edge_input_dummy_size , n_conv_blocks = 3, nfilts = 18, nfilts2 = 50, ksize = 3, output_size = None, use_dropout = True, activation_type = 'leaky_relu', channels_in = 2):
        """
        A simple 1D CNN for extracting features from the timeseries. It is used in the graph_independent graphnet block. 
        Each conv block is as such:
         * 1Dconv kernelsize/stride/filters : 1 / 1 / nfilts2 (e.g. 50)
         * 1Dconv kernelsize/stride/filters : 2 / 2 / nfilts  (e.g. 18)
         * 1Dconv kernelsize/stride/filters : 2 / 2 / nfilts  (e.g. 18)
         * (optional) dropout(0.2)
         * activation
         * 1Dconv kernelsize/stride/filters : 2 / 2 / nfilts  (e.g. 18)
         * AveragePooling(kernel = 2)

         The network returned is `n_conv_blocks' of the aformentioned stacked. 

        parameters:
            n_conv_blocks : number of convolutional blocks stacked.
            nfilts        : number of bottleneck filts (for instance 18)
            nfilts2       : number of filters for the 1x1 convolution (typically larger than nfilts)
            ksize         : size of kernel used for all internal convs (3)
            output_size   : the node state size (default: None)
            use_dropout   : use/notuse dropout between conv layers (some literature suggests it does not help)
            activation    : the activation used after the dropout layer.

          edge_input_dummy_size : This has to do with the implementation of the node block. For uniform treatment of edge inputs, 
        """
        txt2act = {'relu' : tf.keras.layers.ReLU(), 'leaky_relu' : tf.keras.layers.LeakyReLU()}
        _activation = lambda: txt2act[activation_type]


        xin_node_ts = tf.keras.Input(shape = (None, channels_in) , name = "timeseries_input"); 
        xin_edge_dummy = tf.keras.Input(shape = ( edge_input_dummy_size), name = "edge_input_dummy");

        def conv_block(conv_block_input, names_suffix= ""):
            yout_ = Conv1D(kernel_size = 1 ,  filters = nfilts2, strides = 1, use_bias= False,name = "conv_fcnA"+names_suffix)(conv_block_input)
            yout_ = Conv1D(kernel_size=ksize, filters = nfilts, strides=2  , use_bias= False,name  = "conv_fcnB"+names_suffix)(yout_)
            yout_ = Conv1D(kernel_size=ksize, filters = nfilts, strides=2  , use_bias= False,name  = "conv_fcnC"+names_suffix)(yout_)
            if use_dropout:
                yout_ = Dropout(rate = 0.2)(yout_)
            yout_ = Conv1D(kernel_size=ksize,strides=2, filters = nfilts2,use_bias= True)(yout_)
            yout_ = _activation()(yout_)
            #yout_ = keras.layers.AveragePooling1D(pool_size=2)(yout_)
            return yout_
        
        yout = conv_block(xin_node_ts)
        yout = keras.layers.AveragePooling1D(pool_size=2)(yout)
        for b in range(n_conv_blocks-1):
            yout = conv_block(yout, names_suffix=str(b))
        

        yout = keras.layers.GlobalAveragePooling1D()(yout)
        #yout = keras.layers.GlobalMaxPooling1D()(yout)
        yout = Dense(output_size, use_bias = True)(yout)
        yout = keras.layers.LayerNormalization()(yout)
        yout = tf.keras.layers.LeakyReLU()(yout)

        mconv = keras.Model(inputs = [xin_edge_dummy,xin_node_ts], outputs = yout)
        return mconv
    
    def make_graphnet_comp_blocks(self, n_node_state_input_gi = None):
        """
        Prepares the graphnet blocks for the subsequent computation. 
        Subsequently these blocks are composed so that a series of inputs can return
        a gamma distribution directly.
        """
        #NETWORK_STATES_SIZE = 30
        n_node_state_input , n_edge_state_input = [self.edge_and_node_state_size,self.edge_and_node_state_size]
        n_edge_output = n_edge_state_input
        
        batch_size = 10; # An arbitrary number, to create a batch and call the 
                         #functions once to initialize them.

        n_edge_state_input_gi = 1
        n_edge_output_gi = self.edge_and_node_state_size;
        
        ##########################################
        # Graph independent processing:
        edge_mlp_gi = self.make_edge_function_gi(n_edge_state_input = n_edge_state_input_gi,
                                            n_edge_state_output= n_edge_output_gi,
                                            n_node_state_input = n_node_state_input_gi)

        conv_head_params = self.cnn_params
        conv_head_params.update({'edge_input_dummy_size' : n_edge_state_input_gi, 'output_size' : n_node_state_input })
        node_mlp_gi = self.make_conv_input_head_node_function(**conv_head_params ) #edge_input_dummy_size=n_edge_state_input_gi, output_size = n_node_state_input)

        node_mlp_gi([np.random.randn(batch_size,n_edge_state_input_gi),np.random.randn(batch_size,n_node_state_input_gi,self.cnn_params['channels_in'])])
        
        graph_indep = GraphNet(edge_function = edge_mlp_gi,
                               node_function = node_mlp_gi,
                               edge_aggregation_function= None, 
                               node_to_prob= None)

        #########################################
        # Graph processing:
        
        edge_mlp = self.make_edge_function(n_edge_state_input,n_edge_output, n_node_state_input) # THe node state is used for two nodes.
        dat_list= [vv.astype("float32") for vv in [np.random.randn(batch_size,n_edge_state_input), np.random.randn(batch_size,n_node_state_input), np.random.randn(batch_size,n_node_state_input)]]
        edge_mlp(dat_list)

        node_mlp = self.make_node_function(n_edge_state_input, n_node_state_input)
        node_to_prob_mlp = self.make_gamma_node_observation_mlp(n_node_state_input);
        node_to_prob_mlp(np.random.randn(batch_size,n_node_state_input))
        node_mlp([vv.astype("float32") for vv in [np.random.randn(batch_size,n_edge_state_input), np.random.randn(batch_size,n_node_state_input)]])
        per_node_edge_aggregator = self.make_edge_aggregation_function(n_edge_output)
        edge_aggregation_function = per_node_edge_aggregator

        gn = GraphNet(edge_function = edge_mlp,
                      node_function=node_mlp,
                      edge_aggregation_function=edge_aggregation_function,
                      node_to_prob= node_to_prob_mlp)
        self.core = gn
        self.graph_indep = graph_indep

    
        
    def eval_graphnets(self,graph_data_, iterations = 5, eval_mode = "batched", return_reparametrization = False,return_final_node = False, return_intermediate_graphs = False, node_index_to_use = -1):
        """
        graph_data_                : is a "graph" object that contains a batch of graphs (more correctly, a graph tuple as DM calls it)
        iterations                 : number of core iterations for the computation.
        eval_mode                  : "batched" (batch nodes and edges before evaluation) or "safe" (more memory efficient - less prone to OOM errors no batching).-
        return_distr_params        : return the distribution parameters instead of the distribution itself. This is in place because of 
                                     some buggy model loading (loaded models don't return distribution objects).
        return_intermediate_graphs : Return all the intermediate computations.
        """
        graph_out = self.graph_indep.graph_eval(graph_data_,eval_mode = eval_mode)
        intermediate_graphs = [];
        for iterations in range(iterations):
            graph_out = self.core.graph_eval(graph_out, eval_mode = eval_mode) + graph_out # Addition adds all representations (look at implementation of "Graph")
            if return_intermediate_graphs:
                intermediate_graphs.append(graph_out.copy())

        if return_intermediate_graphs:
            return intermediate_graphs

        # Finally the node_to_prob returns a reparametrized "Gamma" distribution from only the final node state
        node_final = graph_out.nodes[node_index_to_use].node_attr_tensor
        if return_final_node:
            return node_final

        if not return_reparametrization:
            return self.core.node_to_prob_function(node_final)
        else:
            return self.core.node_to_prob_function.get_layer("output")(node_final)

    def bootstrap_eval_graphnets(self, graph_data_, iterations = 5,n_bootstrap_samples = 1, n_nodes_keep = 5, eval_mode = "batched", return_final_node = False, node_index_to_use = -1):
        """
        Evaluate multiple random samples of nodes from the past. 
        The signature is alsmost the same as `eval_graphnets` with the difference of the parameters n_boostrap_samples (how many times to resample the past nodes) and n_nodes_keep 
        (how many nodes from the past to keep). The last node is always in the computed sample.
        """
        bootstrap_results = [];
        for nbs in range(n_bootstrap_samples):

            keep_nodes = [graph_data_.nodes[-1]]
            node_indices = list(np.random.choice(len(graph_data_.nodes)-1,n_nodes_keep-1, replace = False))
            node_indices.sort()
            node_indices.append(len(graph_data_.nodes)-1)
            subgraph = graph_data_.get_subgraph_from_nodes([graph_data_.nodes[i] for i in node_indices]) # are gradients passing?
            bootstrap_result = self.eval_graphnets(subgraph, iterations = iterations, eval_mode = eval_mode, return_final_node = True, node_index_to_use = -1)
            bootstrap_results.append(bootstrap_result)

        if return_final_node is False:
            bootstrap_node_value = tf.reduce_mean(bootstrap_results,0)
            return self.core.node_to_prob_function(bootstrap_node_value)
        else:
            return bootstrap_results

        return bootstrap_results
          

    def set_weights(self,weights):
        """
        Takes a list of weights (as returned from a similar object)  and sets the to the functions of this one.
        """
        for w , new_weight in zip([*self.core.weights(), *self.graph_indep.weights()][:] , new_weights):
            w = new_weight





class GraphNet:
    """
    Input is a graph and output is a graph.
    Encapsulates a GraphNet computation iteration.
    
    Supports model loading and saving (for a single GraphNet)

    Should treat the situations where edge functions do not exist more uniformly.
    Also there is no Special treatment for "globals".
    """
    def __init__(self, edge_function = None, node_function = None, edge_aggregation_function = None, segment_indexed_edge_aggregation_function = None, node_to_prob= None, graph_independent = False):
        """
        parameters:
            edge_function             : the edge function (depending on whether the graphnet block is graph 
                                        independent, and whether the source and destinations are used,
                                        this has different input sizes)
            node_function             : the node function (if this is graph independent it has only node inputs)
            edge_aggregation_function : the edge aggregation function used in the non-fully batched evaluation 
                                        modes. ("batched" and "safe")
           segment_indexed_edge_aggregation_function : An edge aggregation function that uses segment indexing
                                        for better performance while performing batched computation (the edges 
                                        are not explicitly iterated for aggregation). This is used when 
                                        working with GraphTuples (as compared to single "graph" objects that 
                                        have graphs with the same connectivity in each batch).
           node_to_prob               : the function that takes the final graph and returns a 

        """
        self.segment_indexed_edge_aggregation_function = segment_indexed_edge_aggregation_function
        self.edge_function             = edge_function
        self.node_function             = node_function
        self.edge_aggregation_function = edge_aggregation_function        
        self.node_to_prob_function = node_to_prob
        self.graph_independent = graph_independent
        # Needed to treat the case of no edges.
        # If there are no edges, the aggregated edge state is zero.
        
        if self.edge_function is not None: # a messy hack:
            self.edge_input_size = self.edge_function.inputs[0].shape[1] # input dimension 1 of edge mlp is the edge state size by convention.

    @staticmethod
    def make_from_path(path):
        graph_functions = GraphNet.load_graph_functions(path)
        return GraphNet(**graph_functions)


    def get_graphnet_input_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update({"edge_function"  : [i.shape for i in self.edge_function.inputs]})
        result.update({"node_function":[i.shape for i in self.node_function.inputs]})
        return result

    def get_graphnet_output_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update({"edge_function"  : [i.shape for i in self.edge_function.outputs]})
        result.update({"node_function":[i.shape for i in self.node_function.outputs]})
        return result

    def get_graphnet_output_shapes(self):
        return [i.shape for i in self.edge_function.outputs] , [i.shape for i in self.node_function.outputs]


        
    def weights(self):
        all_weights = [ *self.edge_function.weights, *self.node_function.weights]
        if self.node_to_prob_function is not None:
            all_weights.extend(self.node_to_prob_function.weights)
        
        if self.edge_aggregation_function is not None and not isinstance(self.edge_aggregation_function, type(tf.reduce_mean)):
            all_weights.extend(self.edge_aggregation_function.weights)
            
        return all_weights
    
    def observe_nodes(self, graph):
        probs = [];
        for n in graph.nodes:
            probs.append(self.node_to_prob_function(n.node_attr_tensor))
            
        return probs
        
    def observe_node(self, node):
        self.node_to_prob_function(node)

    def eval_graph_tuple(self, graph_tuple):
        edges = self.edge_function(graph_tuple.edges)
        nodes = self.node_function(graph_tuple.nodes)
    def __call__(self, graph):
        return self.graph_eval(graph)

    def graph_tuple_eval(self,tf_graph_tuple):
        # This method parallels what the deepmind library does for faster batched computation. 
        # The `tf_graph_tuple` contains edge, nodes, n_edges, n_nodes, senders (indices), receivers (indices).
        # * the "edges" and "nodes" are already stacked into a tensor
        # * If .from_node or .to_node tensors are needed for the edge computations they are gathered according to the senders and receivers tensors.
        # * the edge function is applied to edges
        # * the edge function outputs are aggregated according to the "receivers" tensor to yield the messages.
        # 
        None

    def graph_eval(self, graph, eval_mode = "batched"):
        # Evaluate all edge functions:
        self.eval_edge_functions(graph, eval_mode = eval_mode)

        batch_size             = graph.nodes[0].shape[0]; # This will be related to the input graph tuple.

        edge_input_size = self.edge_input_size ; # This relates to the graphnet being evaluated.

        # Aggregate edges per node:
        edge_to_node_agg_dummy = np.zeros([batch_size, edge_input_size]);

        # Compute the edge-aggregated messages:
        edge_agg_messages_batch = []
        node_agg_messages_batch = []
        for n in graph.nodes: # this explicit iteration is expensive and unnecessary in most cases. The DM approach (graph tuples) seems better - do that.
            if len(n.incoming_edges) is not 0:
                if self.edge_aggregation_function is not None:
                    #print([e.edge_tensor for e in n.incoming_edges])
                    edge_vals_ = tf.stack([e.edge_tensor for e in n.incoming_edges])
                    #print(edge_vals)
                    edge_to_node_agg = self.edge_aggregation_function(edge_vals_)
                else:
                    edge_to_node_agg = edge_to_node_agg_dummy
            else:
                edge_to_node_agg = edge_to_node_agg_dummy

            #Inside the loop!
            if eval_mode == 'safe':
                if self.graph_independent:
                    node_attr_tensor = self.node_function([n.node_attr_tensor])
                    n.set_tensor(node_attr_tensor)
                else:
                    node_attr_tensor = self.node_function([edge_to_node_agg, n.node_attr_tensor])
                    n.set_tensor(node_attr_tensor)

            if eval_mode == 'batched':
                edge_agg_messages_batch.append(edge_to_node_agg)
                node_agg_messages_batch.append(n.node_attr_tensor)

        if eval_mode == 'batched':
            # If we compute in batched mode, there is some reshaping to be done in the end.
            node_input_shape = graph.nodes[0].shape # nodes and edges (therefore graphs as well) could contain multiple datapoints. This is to treat this case.
            node_output_shape =self.node_function.output.shape

            nodes_agg_messages_concat = tf.concat(node_agg_messages_batch,axis = 0)
            edges_agg_messages_concat = tf.concat(edge_agg_messages_batch, axis = 0)


            if self.graph_independent:
                batch_res = self.node_function(
                        [nodes_agg_messages_concat])

            else:
                batch_res = self.node_function(
                        [edges_agg_messages_concat, nodes_agg_messages_concat])

            unstacked = tf.unstack(
                    tf.reshape(
                        batch_res,[-1,*node_input_shape[0:1],*node_output_shape[1:]]), axis = 0
                    )
            for n, nvalue in zip(graph.nodes, unstacked):
                n.set_tensor(nvalue)

        return graph

    def save(self, path):
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        path_labels = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        for model_fcn, label in zip(functions, path_labels):
            if model_fcn is not None:
                d_ = os.path.join(path,label)
                model_fcn.save(d_)
                
    @staticmethod
    def load_graph_functions(path):
        """
        Returns a list of loaded graph functions.
        """
        function_rel_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        functions = {};

        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        avail_functions = os.listdir(path) # the path should have appropriately named folders that correspond to the diffferent graph functions. All are keras models.
        for l in function_rel_paths:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                
            else:
                model_fcn = tf.keras.models.load_model(d_)

                functions.update({l:model_fcn})

                print("loading %s"%(d_))
        return functions




    def load(self, path):
        """
        Load a model from disk. If the model is already initialized the current graphnet functions are simply overwritten. 
        If the model is un-initialized, this is called from a static method (factory method) to make a new object with consistent properties.

        """
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        all_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        path_label_to_function = {z:v for z,v in zip(all_paths,functions)}
        path_labels = os.listdir(path) #
        
        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        for l in path_labels:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                next
            else:
                model_fcn = tf.keras.models.load_model(d_)
                path_label_to_function[l] = model_fcn
                print(path_label_to_function[l] )

                print("loading %s"%(d_))

            
    def eval_edge_functions(self,graph, eval_mode = "batched"):
        """
        Evaluate all edge functions. Batched mode has some shape juggling going on.
        If you see weird behaviour that's the first place to look (tests not written yet. :totest:)
        
        params:
          graph     - the graph containing the edges 
          eval_mode - "safe" or "batched" (batched is also safe if state shapes are respected)
        """
        assert(eval_mode in ['safe', 'batched'])
        if len(graph.edges) == 0:
            return 
        
        if self.edge_aggregation_function is None: # this happens in graph-independent networks (there is no aggregation and no message passing)
            if eval_mode == 'safe':
                for edge in graph.edges:
                    edge_tensor = self.edge_function([edge.edge_tensor])
                    edge.set_tensor(edge_tensor)
                    
            if eval_mode == 'batched':
                edges_ = graph.edges
                edges_shape = edges_[0].shape
                edges_concat = tf.concat([e.edge_tensor for e in edges_],axis = 0)
                batch_res = self.edge_function([edges_concat])
                unstacked = tf.unstack(tf.transpose(tf.reshape(batch_res,[-1,*edges_shape[0:1],*batch_res.shape[1:]]),[0,1,2]), axis = 0)
                for e, evalue in zip(edges_, unstacked):
                    e.set_tensor(evalue)

                
        else:
            if eval_mode == 'safe':
                for edge in graph.edges:
                    edge_tensor = self.edge_function([edge.edge_tensor, edge.node_from.node_attr_tensor, edge.node_to.node_attr_tensor])
                    edge.set_tensor(edge_tensor)
                    
            if eval_mode == 'batched':
                edges_ = graph.edges
                edge_function = self.edge_function
                edges_shape = edges_[0].shape
                
                edges_concat = tf.concat([e.edge_tensor for e in edges_],axis = 0)
                node_from_concat = tf.concat([e.node_from.node_attr_tensor for e in edges_], axis = 0)
                node_to_concat= tf.concat([e.node_to.node_attr_tensor for e in edges_],axis = 0)

                
                #         inps = { 'edge_state': edges_concat, 'node_sender' : node_from_concat, 'node_receiver' : node_to_inputs}
                #         res = edge_function(inps)
                batch_res = self.edge_function([edges_concat, node_from_concat, node_to_concat])
                unstacked = tf.unstack(tf.transpose(tf.reshape(batch_res,[-1,*edges_shape[0:1],*batch_res.shape[1:]]),[0,1,2]), axis = 0)
                for e, evalue in zip(edges_, unstacked):
                    e.set_tensor(evalue)

                
def make_mlp(units, input_tensor_list , output_shape):
    """
    A default method for making a small MLP:
    """
    if len(input_tensor_list) > 1:
        edge_function_input = keras.layers.concatenate(input_tensor_list);
    else:
        edge_function_input = input_tensor_list[0] #essentialy a list of a single tensor.
    #print(units, edge_function_input)

    y = Dense(units)(  edge_function_input)
    y=  Dropout(rate = 0.2)(y)
    y = Dense(units, activation = "relu")(y)
    y=  Dropout(rate = 0.2)(y)
    y = Dense(units, activation = "relu")(y)
    y = Dense(output_shape[0])(y)
    return tf.keras.Model(inputs = input_tensor_list, outputs = y)


def make_node_mlp(units,
        edge_state_input_shape = None,
        node_state_input_shape = None, 
        node_emb_size= None,
        graph_indep = False):

    node_state_in = Input(shape = node_state_input_shape, name = "node_state");

    if graph_indep:
        return make_mlp(units, [node_state_in], node_emb_size)

    if not graph_indep:
        agg_edge_state_in = Input(shape = edge_state_input_shape, name = "edge_state_agg");
        return make_mlp(units, [agg_edge_state_in, node_state_in],node_emb_size)


def make_edge_mlp(units,
        edge_state_input_shape = None,
        edge_state_output_shape = None, 
        sender_node_state_output_shape = None,
        receiver_node_state_shape = None,
        edge_output_state_size = None,
        use_edge_state = True,
        use_sender_out_state = True,
        use_receiver_state = True,
        graph_indep = False):
    """
    When this is a graph independent edge function, the input is different: 
    The node states from the sender and receiver are not used!
    """
    tensor_in_list = []
    if use_edge_state:
        edge_state_in = Input(
                shape = edge_state_input_shape, name = "edge_state");
        tensor_in_list.append(edge_state_in)

    if use_sender_out_state:
        node_state_sender_out = Input(
                shape = sender_node_state_output_shape, name = "sender_node_state");
        tensor_in_list.append(node_state_sender_out)

    if use_receiver_state:
        node_state_receiver_in = Input(
                shape = receiver_node_state_shape, name = "receiver_node_state");

    if graph_indep:
        try:
            assert(use_sender_out_state is False)
            assert(use_receiver_out_state is False)
        except:
            ValueError("The receiver and sender nodes for graph-independent blocks should not be used! It was attempted to create an edge function for a graph-indep. block containing receiver and sender states as inputs.")
        ## Building the edge MLP:
        tensor_input_list = [edge_state_in]
        return make_mlp(units,tensor_input_list,edge_output_state_size )  
    else:
        ## Building the edge MLP:
        return make_mlp(units,tensor_in_list,edge_output_state_size )  

def make_mean_agg(input_size):
    """
    For consistency I'm making this a keras model as well.
    """
    x = Input(shape = input_size)
    y = tf.reduce_mean(x,0)
    return tf.keras.Model(inputs = x, outputs = y)

def make_mlp_graphnet_functions(units,
        input_size, output_size, 
        graph_indep = False, message_size = None):
    """
    Make the 3 functions that define the graph (no Nodes to Global and Edges to Global)
    It is assumed that all inputs and all outputs are the same size for nodes and edges.
    """
    edge_input = input_size
    node_input = input_size
    edge_output = output_size
    node_output = output_size
    if message_size is None:
        message_size = output_size

    edge_mlp_args = {"edge_state_input_shape" : (edge_input,),
            "edge_state_output_shape" : (message_size,),
            "sender_node_state_output_shape" : (node_input,), # this has to be compatible with the output of the aggregator.
            "edge_output_state_size":(edge_output,),
            "receiver_node_state_shape" : (node_input,)} 
    edge_mlp_args.update({"graph_indep" : graph_indep})
    edge_mlp = make_edge_mlp(units, **edge_mlp_args)

    if not graph_indep:
        agg_fcn  = make_mean_agg([None, *edge_mlp.outputs[0].shape[1:]]) # First dimension - incoming edge index
    else:
        #edge_mlp = None
        agg_fcn = None

    node_mlp_args = {"edge_state_input_shape": (input_size,),
            "node_state_input_shape" : (input_size,),
            "node_emb_size" : (output_size,)}
    node_mlp_args.update({"graph_indep" : graph_indep})

    node_mlp = make_node_mlp(units, **node_mlp_args)
    return {"edge_function" : edge_mlp, "node_function": node_mlp, "edge_aggregation_function": agg_fcn} # Can be passed directly  to the GraphNet construction with **kwargs
