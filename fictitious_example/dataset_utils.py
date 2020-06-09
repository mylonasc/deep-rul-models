import numpy as np
from .minigraphnets import Node, Edge, Graph


def lhs_sample(npoints, range_size = 10,  lranges=None):
    if lranges is None:
        lranges = npoints;

    nchoices = np.random.choice(range_size,npoints);
    ret_vals = [];
    for k in range(lranges) :
        ret_vals.append( k * range_size + nchoices[k])
    return ret_vals

def transform_exp_data_to_random_signal_params(exp_dat_, rr = None):
    """
    Splits the stochastic degradation to segments that are to be the inputs to another
    function that creates samples from 
    rr: how many consecutive samples to use for making a segment.
    """
    ee = exp_dat_[:-(exp_dat_.shape[0]%rr)].reshape([-1,rr]) if (exp_dat_.shape[0]%rr != 0) else exp_dat_.reshape([-1,rr])
    return ee

def add_disturbances_in_signal(ee, speed =50, npoints = 1000):
    # Have disturbances that are proportional to the gamma random variable.
    #  points per segment.
     # a parameter in order to make the problem a bit more difficult.
    t = np.linspace(0,2*np.pi,npoints); # segment time

    v = np.sin(t*speed)*0.1 + np.random.randn(t.shape[0])*0.0
    
    #ndist_samples = 10
    ndisturb_samples = 20
    tseg = np.linspace(0,2*np.pi,ndisturb_samples)

    random_segment_pos = lhs_sample(ee.shape[1],int(np.floor(npoints/ee.shape[1]))) # controls the index where the signal disturbance is added.
    
    #pplot.plot(np.sin(v))
    vals=[]
    for seg in ee:
        vc = v.copy();
        random_segment_pos = lhs_sample(10,int(npoints/10)+1)
        for t_s,s in zip(random_segment_pos, seg):
            vc[t_s:t_s+ndisturb_samples] += np.sin(t[t_s:t_s+ndisturb_samples]*speed*5)*(s/160+1)**2

        vals.append(vc)
            #vals = t_s
            #v[t_] + np.sin(t*)
            #pplot.vlines(tt, ymin = 0 ,ymax = 0.2)
    len(vals)
    return np.vstack(vals)


def get_signal_for_segments(exp_dat_, speed, rr=10) :
    return add_disturbances_in_signal(transform_exp_data_to_random_signal_params(exp_dat_, rr = rr), speed)


def get_dat(d, rr = None):
    """
    Transforms the latent values to a timeseries segment. 
    
    The timeseries segment is used for prediction. The "case" input parameter contains the information of which loading case we are producing datra from.
    Different cases have a different background "noise". The network should learn to exploit this background for better prediction (different cases have different rates of damage accumulation).
    """
    speed_dict = {0:20, 1:25, 2:30};
    a = d['latent_values']
    get_signal_for_segments = lambda exp_dat_, speed : add_disturbances_in_signal(transform_exp_data_to_random_signal_params(exp_dat_, rr =rr), speed)

    l = d['latent_values']
    l2 =d['ttf']
    trunc_l = l[:-(l2.shape[0]%rr)] if (l2.shape[0]%rr !=0)  else l
    y = np.mean(trunc_l.reshape([-1,rr]),1)
    case = d['case']
    X = get_signal_for_segments(l, speed= speed_dict[case])
    eid = d['exp_index']
    exp_data = {"X" : X , "eid" : eid , "y" : y, "case" : case}    
    return exp_data


def get_graph_training_data(dat, nsamples):
    rand_1 = np.random.choice(int(dat['X'].shape[0]/3),nsamples)
    rand_2 = np.random.choice(int(dat['X'].shape[0]/3),nsamples) + int(dat['X'].shape[0]/3)
    rand_3 = np.random.choice(int(dat['X'].shape[0]/3),nsamples) + int(dat['X'].shape[0]/3) * 2
    X_n1, X_n2, X_n3 = [dat['X'][r_][..., np.newaxis] for r_ in [rand_1, rand_2, rand_3]]
    t_n1, t_n2, t_n3 = [dat['y'] [r_] for r_ in [rand_1, rand_2, rand_3]]
    dt_12 = t_n2 - t_n1
    dt_13 = t_n3 - t_n1
    dt_23 = t_n3 - t_n2
    n1 = Node(X_n1)
    n2 = Node(X_n2)
    n3  = Node(X_n3)
    e12 = Edge(dt_12.reshape([nsamples , 1]),n1,n2)
    e13 = Edge(dt_13.reshape([nsamples , 1]), n1,n3)
    e23 = Edge(dt_23.reshape([nsamples , 1]), n2,n3)
    graphs = Graph([n1,n2,n3], [e12,e13,e23])
    outputs = t_n3
    return graphs, outputs


def get_graph_training_data_multiple_experiments(all_dat_, nsamples_per_exp=100, rr = 10):
    all_graphs = []
    all_outputs = []
    for d in all_dat_:
        exp_dat = get_dat(d, rr )
        
        gout, outs = get_graph_training_data(exp_dat, nsamples_per_exp)
        all_graphs.append(gout)
        all_outputs.append(outs)
        
    return all_graphs, all_outputs




###        
# Had an implementation in another file. Keeping this here just in case there is sth different I haven't noticed.
##

#""" Classes for basic manipulation of GraphNet """
#class Node:
#    def __init__(self, node_attr_tensor):
#        self.node_attr_tensor = node_attr_tensor
#        self.incoming_edges = [];
#        self.shape = self.node_attr_tensor.shape
#        
#    def get_state(self):
#        return self.node_attr_tensor
#    def set_tensor(self, tensor):
#        self.node_attr_tensor = tensor
#        self.shape = self.shape = tensor.shape
#        
#    def copy(self):
#        node_attr_tensor = self.node_attr_tensor
#    
#    
#    
## My implementation relies on eager mode and all computation happens in place. In reality only nodes
## and edges have data and the graph class is just for defining the computation between them.
#class Edge:
#    def __init__(self, edge_attr_tensor, node_from, node_to):
#        self.edge_tensor = edge_attr_tensor
#        self.node_from = node_from
#        self.node_to = node_to
#        self.shape = self.edge_tensor.shape
#        
#        # Keep a reference to this edge since it is needed for aggregation afterwards.
#        node_to.incoming_edges.append(self)
#    def set_tensor(self, edge_tensor):
#        self.edge_tensor = edge_tensor
#        self.shape = edge_tensor.shape
#    
#    def copy(self, nodes_correspondence):
#        edge_tensor = self.edge_tensor
#        node_from = nodes_correspondence[self.node_from]
#        node_to   = nodes_correspondence[self.node_to]
#        return Edge(edge_tensor, node_from, node_to)
#        
#class Graph:
#    def __init__(self, nodes, edges):
#        self.nodes = nodes
#        self.edges = edges
#        
#    def copy(self):
#        # copy attributes of nodes and edges and re-create graph connectivity:
#        nodes_coppied = [Node(n.node_attr_tensor.copy()) for n in self.nodes]
#        nodes_correspondence = {s:c for s , c in zip(self.nodes,nodes_coppied)}
#        # Instantiate the new edges:
#        coppied_edge_instances = []
#        for e in self.edges:
#            enew = Edge(e.edge_tensor, nodes_correspondence[e.node_from], nodes_correspondence[e.node_to])
#            coppied_edge_instances.append(enew)
#        return Graph(nodes_coppied, coppied_edge_instances)
        
