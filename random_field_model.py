from minigraphnets import Graph, Node, Edge
import numpy as np
import tqdm




def sample_graph_from_2d_result(points, all_vals_, pct_x = 10, pct_y = 10, cutoff = 0.5):
    u = np.random.uniform(all_vals_) # split data randomly and exclusively in two sets (source and target datapoints)
    my = (u <=  (pct_x/100 + pct_y/100)) * (u>pct_x/100)
    mx = (u <= (pct_x/100 + pct_y/100))  * (u<=pct_x/100)
    
    points_x , points_y= [points[m_] for m_ in [mx,my]]
    vals_x, vals_y = [all_vals_[m_] for m_ in [mx, my]]
    # make a graph out of this:
    # Input graph has zero at the vals_y and edges with source nodes the "points_x" and destination "points_y".
    input_node_features = np.hstack([vals_x, np.zeros_like(vals_y)])
    
    all_points = np.vstack([points_x, points_y])
    all_nodes = []
    all_edges = []
    
    all_nodes_output = []
    tot_point_positions = []
    for p1, source_node_val in zip(points_x, vals_x):
        source_node = Node(np.array([[source_node_val]]))
        tot_point_positions.append(p1)
        all_nodes_output.append(source_node)
        all_nodes.append(source_node)
        # Make bi-directional edges for the points that are observed:
        for p2, dest_node_val in zip(points_x, vals_x):
            if np.linalg.norm(p2 - p1) < cutoff:
                dest_node = Node(np.array([[dest_node_val]]))
                
                all_nodes.append(dest_node)
                all_nodes_output.append(dest_node)
                tot_point_positions.append(p2)
                
                
                d = np.array([[np.sum((p1 - p2)**2)]])
                all_edges.append(Edge(node_from=source_node, node_to=dest_node , edge_attr_tensor=d ))
                all_edges.append(Edge(node_from = dest_node, node_to = source_node , edge_attr_tensor = d))
            
        # make uni-directional edges for the unobserved points
        for p2, dest_node_val in zip(points_y, vals_y):
            if np.linalg.norm(p2 - p1) < cutoff:
                dest_node_input = Node(np.array([[dest_node_val * 0]]))
                dest_node_output = Node(np.array([[dest_node_val]]))
                tot_point_positions.append(p2)
                
                all_nodes.append(dest_node_input)
                all_nodes_output.append(dest_node_output)

                d = np.array([[np.sum((p1 - p2)**2)]])
                all_edges.append(Edge(node_from=source_node, node_to=dest_node_input , edge_attr_tensor=d ))
            
    input_graph = Graph( all_nodes, all_edges)
    output_graph = Graph( all_nodes_output, [])
    return input_graph, output_graph, tot_point_positions


class ExpQuadKernel:
    def __init__(self,l):
        """
        An exponentiated quadradic kernel function.
        """
        self.l = l
    
    def kernel_function(self, t,s):
        return np.exp(-(np.linalg.norm(t-s)**2)/self.l**2)
    
    def get_kernel(self, points):
        K = np.zeros([points.shape[0], points.shape[0]])
        for i,p in enumerate(points):
            for j,q in enumerate(points):
                if i>=j:
                    K[i,j] = self.kernel_function(p,q)
                
        K = K + K.T-np.diag(np.diag(K))+np.eye(K.shape[0])*0.001
        #K = K + np.eye(K.shape[0])*np.min(K)*1e-20
        return K
    
    def get_chol(self, points):
        K = self.get_kernel(points)
        return np.linalg.cholesky(K)
    
def get_multiple_graph_samples_random_fields(nsamples, vals, all_points, pct_observed=10, pct_unobserved=None , cutoff = 0.5):
    """
    Returns graph samples from a random field.

    nsamples: how many samples per random field to take
    vals: the values of the random field.
    all_points: the coordinates of all the points of the random fields
    pct_observed: the percentage of the positions of the random field assumed observed for each sample
    pct_unobserved: the percentage of the positions of the random field assumed unobserved.


    """
    if pct_unobserved is None:
        pct_unobserved = pct_observed
    
    input_graphs = []
    output_graphs = []
    node_pos = []
    for v in vals:
        for sample in range(nsamples):
            in_graph, out_graph , node_positions = sample_graph_from_2d_result(all_points, v,pct_x = pct_observed, pct_y = pct_unobserved, cutoff = cutoff)
            input_graphs.append(in_graph)
            output_graphs.append(out_graph)
            node_pos.append(node_positions)
        print(".\r")



    return input_graphs, output_graphs , node_pos

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf

import matplotlib.pyplot as pplot


if __name__ == "__main__":

    ## Creation of the random field data:
    # Creates a 20x20 grid of points and a random field with them.
    k = ExpQuadKernel(0.3) 
    #points = np.random.randn(10,2)
    npoints = 20
    [xx,yy] = np.meshgrid(np.linspace(-1,1,npoints), np.linspace(-31,1,npoints))
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    C = k.get_chol(points)

    r = np.random.randn(np.prod(xx.flatten().shape))
    all_vals = [];
    for i in range(2):
        vals = r @ C.T
        r = np.random.randn(np.prod(xx.flatten().shape))
        all_vals.append(vals) #vals.reshape(xx.shape))

    # samples a single graph from 1 of the sampled random fields:
    # in_graph , out_graph = sample_graph_from_2d_result(points, all_vals[0])
    input_graphs, output_graphs  = get_multiple_graph_samples_random_fields(5,all_vals,points)

    #graph_state_size = (10,);
    graph_state_size = 32;
    units = 32

    qoi_size = 1; # dimension of the random field.

    functions_encode = make_mlp_graphnet_functions(units,input_size=qui_size, output_size=graph_state_size, graph_indep=True)
    functions_core   = make_mlp_graphnet_functions(units, input_size = graph_state_size,output_size = graph_state_size)
    functions_decode = make_mlp_graphnet_functions(units ,input_size = graph_state_size,output_size=1, graph_indep = True)

    gn_encode = GraphNet(**functions_encode)
    gn_process = GraphNet(**functions_core)
    gn_decode = GraphNet(**functions_decode)

    def eval_network(graph, core_iterations = 3, residual = True, eval_mode = "batched"):
        g = gn_encode.graph_eval(graph.copy(), eval_mode = eval_mode)
        for k in range(core_iterations):
            if residual:
                g = g + gn_process.graph_eval(g, eval_mode= eval_mode)
            else:
                g = gn_process.graph_eval(g, eval_mode = eval_mode)
                
        return gn_decode.graph_eval(g, eval_mode = eval_mode)

    res = eval_network(input_graphs[0])

    ## train the network (deterministic)


