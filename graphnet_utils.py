import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras import Sequential,Model
import tensorflow.keras as keras

class GraphNetFunctionFactory:
    def __init__(self, NETWORK_SIZE_GLOBAL = 50, USE_PRENETWORKS = True, EDGE_NODE_STATE_SIZE = 15):
        """
        Summary: 
          A factory for graphnet functions. It is custom made for the problems of RUL from time-series. 
          It can be adapted to other prediction models. All models (except the aggregation function) are 
          relatively small MLPs terminated by sigmoid()*tanh() activation (simple tanh could also work).
          
        NETWORK_SIZE_GLOBAL: A parameter controlling the width of different networks involved.
    
        USE_PRENETWORKS:     Use a deeper architecture (see code)
        
        GRAPH_STATE_SIZE:    the size of the node states, and edge states. This is needed
                             to create consistent graph functions. Eventhough here I'm using the same global size,
                             the sizes of edge states and node states can be different.
        """
        self.network_size_global = NETWORK_SIZE_GLOBAL
        self.use_prenetworks = USE_PRENETWORKS
        self.edge_and_node_state_size = EDGE_NODE_STATE_SIZE
    

    def make_gamma_node_observation_mlp(self, n_node_state_output):
        """
        Takes as input a node state and returns a gamma probability distribution
        """
        seq = keras.Sequential()
        NParams= 1;
        NSamples = 100;
        #seq.add(Dense(n_gamma_internal, use_bias = True, activation = "relu", name = "output1"))
        seq.add(Dense(NParams*2, use_bias = False, activation = lambda x : tf.nn.softplus(x),name = "output"));
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
        gi_path = os.path.join(path,"graph_independent")
        core_path = os.path.join(path,"core")


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

        edge_out_gate = Dense(n_edge_state_output, use_bias = False, activation = "sigmoid", name = "edge_gi_fcnA")(edge_out)
        edge_outB = Dense(n_edge_state_output, use_bias = False, activation = "tanh", name = "edge_gi_fcnB")(edge_out)
        edge_out = edge_outB * edge_out_gate 
        edge_mlp = Model(inputs = edge_state_in,outputs = edge_out)
        
        return edge_mlp

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



    def make_conv_input_head_node_function(self,edge_input_dummy_size , nfilts = 18, nfilts2 = 50, ksize = 3, output_size = None):

        xin_node_ts = tf.keras.Input(shape = (None, 2) , name = "timeseries_input"); 
        xin_edge_dummy = tf.keras.Input(shape = ( edge_input_dummy_size), name = "edge_input_dummy");

        yout = Conv1D(kernel_size = 1 ,  filters = nfilts2, strides = 1, use_bias= False,name = "conv_fcnA")(xin_node_ts)
        yout = Conv1D(kernel_size=ksize, filters = nfilts, strides=2  , use_bias= False,name = "conv_fcnB")(yout)
        yout = Conv1D(kernel_size=ksize, filters = nfilts, strides=2  , use_bias= False,name = "conv_fcnC")(yout)
        #yout = Dropout(rate = 0.2)(yout)
        yout = Conv1D(kernel_size=ksize,strides=2, filters = nfilts2,use_bias= True)(yout)
        yout = tf.keras.layers.LeakyReLU()(yout)
        yout = keras.layers.AveragePooling1D(pool_size=2)(yout)

        yout = Conv1D(kernel_size = 1 ,  filters = nfilts2, strides = 1, use_bias= False,name = "conv_fcnA3")(yout)
        yout = Conv1D(kernel_size=ksize, filters = nfilts , strides=2  , use_bias= False,name = "conv_fcnB3")(yout)
        yout = Conv1D(kernel_size=ksize, filters = nfilts , strides=2  , use_bias= False,name = "conv_fcnC3")(yout)
        #yout = Dropout(rate = 0.2)(yout)
        yout = Conv1D(kernel_size=ksize,strides=2, filters = nfilts2,use_bias= True)(yout)
        yout = tf.keras.layers.LeakyReLU()(yout)
        #yout = keras.layers.AveragePooling1D(pool_size=2)(yout)

        yout = Conv1D(kernel_size = 1 ,  filters = nfilts2, strides = 1, use_bias= False,name = "conv_fcnA4")(yout)
        yout = Conv1D(kernel_size=ksize, filters = nfilts , strides=2  , use_bias= False,name = "conv_fcnB4")(yout)
        yout = Conv1D(kernel_size=ksize, filters = nfilts , strides=2  , use_bias= False,name = "conv_fcnC4")(yout)
        #yout = Dropout(rate = 0.2)(yout)
        yout = Conv1D(kernel_size=ksize,strides=2, filters = nfilts2,use_bias= True)(yout)
        yout = tf.keras.layers.LeakyReLU()(yout)
        #yout = keras.layers.AveragePooling1D(pool_size=2)(yout)

        #yout = keras.layers.GlobalAveragePooling1D()(yout)
        yout = keras.layers.GlobalMaxPooling1D()(yout)
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

        node_mlp_gi = self.make_conv_input_head_node_function(edge_input_dummy_size=n_edge_state_input_gi, output_size = n_node_state_input)

        node_mlp_gi([np.random.randn(batch_size,n_edge_state_input_gi),np.random.randn(batch_size,n_node_state_input_gi,2)])
        
        graph_indep = GraphNet(edge_function = edge_mlp_gi,
                               node_function = node_mlp_gi,
                               edge_aggregation_function= None, 
                               node_to_prob_function= None)

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
                      node_to_prob_function= node_to_prob_mlp)
        self.core = gn
        self.graph_indep = graph_indep
        
    def eval_graphnets(self,graph_data_, iterations = 5):
        """
        graph_data_  : is a "graph" object that contains a batch of graphs (more correctly, a graph tuple as DM calls it)
        iterations   : number of core iterations for the computation.
        """
        graph_out = self.graph_indep.graph_eval(graph_data_)
        for iterations in range(iterations):
            graph_out = self.core.graph_eval(graph_out) + graph_out # Addition adds all representations (look at implementation of "Graph")

        # Finally the node_to_prob returns a reparametrized "Gamma" distribution from only the final node state
        return self.core.node_to_prob_function(graph_out.nodes[-1].node_attr_tensor) 


class GraphNet:
    """
    Input is a graph and output is a graph.
    Encapsulates a GraphNet computation iteration.
    
    Supports model loading and saving (for a single GraphNet)
    """
    def __init__(self, edge_function, node_function, edge_aggregation_function, node_to_prob_function):
        self.edge_function             = edge_function
        self.node_function             = node_function
        self.edge_aggregation_function = edge_aggregation_function        
        self.node_to_prob_function = node_to_prob_function
        # Needed to treat the case of no edges.
        # If there are no edges, the aggregated edge state is zero.
        
        self.edge_input_size = self.edge_function.inputs[0].shape[1] # first input of edge mlp is the edge state size by convention.
        
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
        
    def graph_eval(self, graph):
        # Evaluate all edge functions:
        self.eval_edge_functions(graph)
        
        batch_size             = graph.nodes[0].shape[0]; # This will be related to the input graph tuple. 
        
        edge_input_size = self.edge_input_size ; # This relates to the graphnet being evaluated.
        
        # Aggregate edges per node:
        edge_to_node_agg_dummy = np.zeros([batch_size, edge_input_size]);
        
        for n in graph.nodes:
            if len(n.incoming_edges) is not 0:                
                if self.edge_aggregation_function is not None:
                    edge_vals_ = tf.stack([e.edge_tensor for e in n.incoming_edges])
                    edge_to_node_agg = self.edge_aggregation_function(edge_vals_)
                    node_attr_tensor = self.node_function([edge_to_node_agg, n.node_attr_tensor])
                    n.set_tensor(node_attr_tensor)
                else:
                    node_attr_tensor = self.node_function([edge_to_node_agg_dummy,n.node_attr_tensor])
                    n.set_tensor(node_attr_tensor)
                    
            else:
                node_attr_tensor = self.node_function([edge_to_node_agg_dummy, n.node_attr_tensor])
                n.set_tensor(node_attr_tensor)
        
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
                
    def load(self, path):
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        all_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        path_label_to_function = {z:v for z,v in zip(all_paths,functions)}
        path_labels = os.listdir(path) #
        
        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        for l in path_labels:
            d_ = os.path.join(path,l)
            if path is None:
                next
            else:
                model_fcn = tf.keras.models.load_model(d_)
                path_label_to_function[l] = model_fcn
            
           
    def eval_edge_functions(self,graph):
        """
        Evaluate all edge functions
        """
        if self.edge_aggregation_function is None:
            for edge in graph.edges:
                edge_tensor = self.edge_function([edge.edge_tensor])
                edge.set_tensor(edge_tensor)
                
        else:
            for edge in graph.edges:
                edge_tensor = self.edge_function([edge.edge_tensor, edge.node_from.node_attr_tensor, edge.node_to.node_attr_tensor])
                edge.set_tensor(edge_tensor)
                
