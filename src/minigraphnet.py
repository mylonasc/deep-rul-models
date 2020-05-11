from tensorflow import keras
from tensorflow.keras import Dense,Sequential, Input
import tensorflow as tf

n_mlp = 10;
n_edge_state = 15;
n_node_state = 17;

def make_edge_function():
    edge_state_in = Input(shape = (n_edge_state), name = "edge_state");
    node_state_sender_in = Input(shape = (n_node_state), name = "node_sender");
    node_state_receiver_in = Input(shape = (n_node_state), name = "node_receiver");

    ## Building the edge MLP:
    edge_out = keras.layers.concatenate([edge_state_in, node_state_sender_in, node_state_receiver_in])
    
    edge_out = Dense(100, activation = "sigmoid")(edge_out)
    edge_out = Dense(n_edge_state, activation = None)(edge_out)
    
    edge_mlp = Model(inputs = [edge_state_in, node_state_sender_in, node_state_receiver_in ] ,outputs = edge_out)
    
    return edge_mlp

def per_node_edge_aggregator(node_incoming_edges):
    """
    This seems to get the pure edge state.
    I can pass the node state in the edge 
    state if I want to have this as well.
    """
    val  = tf.reduce_mean(tf.stack(node_incoming_edges),0)
    return val

def make_node_function():
    agg_edge_state_in = Input(shape = (n_edge_state), name = "edge_state_agg");
    node_prev_state = Input(shape = (n_node_state), name = "node_sender");

    ## Building the edge MLP:
    node_out = keras.layers.concatenate([agg_edge_state_in, node_prev_state]);
    
    node_out = Dense(100, activation = "sigmoid")(node_out)
    node_out = Dense(n_node_state, activation = None)(node_out)
    
    node_out_model = Model(inputs = [agg_edge_state_in, node_prev_state] ,outputs = node_out)
    return node_out_model

edge_mlp = make_edge_function()
edge_mlp([vv.astype("float32") for vv in [np.random.randn(10,n_edge_state), np.random.randn(10,n_node_state), np.random.randn(10,n_node_state)]])

node_mlp = make_node_function()
node_mlp([vv.astype("float32") for vv in [np.random.randn(10,n_edge_state), np.random.randn(10,n_node_state)]])

class Node:
    def __init__(self, node_attr_tensor):
        self.node_attr_tensor = node_attr_tensor
        self.incoming_edges = [];
        
    def get_state(self):
        return self.node_attr_tensor

class Edge:
    def __init__(self, edge_attr_tensor, node_from, node_to):
        self.edge_tensor = edge_attr_tensor
        self.node_from = node_from
        self.node_to = node_to
        
        # Keep a reference to this edge since it is needed for aggregation afterwards.
        node_to.incoming_edges.append(self)
        
class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        
class GraphNet:
    """
    Input is a graph and output is a graph.
    """
    def __init__(self, edge_function, node_function, edge_aggregation_function):
        self.edge_function             = edge_function
        self.node_function             = node_function
        self.edge_aggregation_function = edge_aggregation_function
        
    def graph_eval(self, graph):
        # Evaluate all edge functions:
        self.eval_edge_functions(graph)
        
        # Aggregate edges per node:
        for n in graph.nodes:
            if len(n.incoming_edges) is not 0:
                edge_to_node_agg = self.edge_aggregation_function([e.edge_tensor for e in n.incoming_edges])
                n.node_attr_tensor = self.node_function([edge_to_node_agg, n.node_attr_tensor])
        
        return 
                
    def eval_edge_functions(self,graph):
        """
        Evaluate all edge functions
        """
        for edge in graph.edges:
            edge.edge_tensor = self.edge_function([edge.edge_tensor, edge.node_from.node_attr_tensor, edge.node_to.node_attr_tensor])
            
        
