import numpy as np
from minigraphnets import Edge,Node, Graph

def get_indices(nsamples, ntotal_inds, nmin_idx = 0, nseq_range = None, nmin_idx_diff = None,npoints_per_seq = None,fixed_spacing_indices = False):
        
    for v,vname in zip([nmin_idx, nseq_range, nmin_idx_diff, npoints_per_seq],['nmin_idx', 'nseq_range', 'nmin_idx_diff', 'npoints_per_seq']):
        if v is None:
            raise ValueError("%s cannot be None!"%vname)
        
    nmax_idx = ntotal_inds - nseq_range;
    sample_indices = [];
    for i in range(nsamples):
        
        if fixed_spacing_indices:
            s_ = np.array([i for i in range(1,nseq_range, nmin_idx_diff)]).astype(int)
            s_ = s_[0:npoints_per_seq]
        else:
            s_ = np.sort(np.random.choice(range(1,nseq_range, nmin_idx_diff), npoints_per_seq, replace = False)).astype(int)
        
        s = np.random.choice(range(nmin_idx,nmax_idx),1) + s_
        sample_indices.append(s)
        
    return np.vstack(sample_indices)
    

def data_from_experiment(eid, X_ = None, eid_oh_ = None, yrem_norm_ = None):
    ids = (np.argmax(eid_oh_,1) == eid)
    Xexp = X_[ids,:,:];
    yrem_exp_ = yrem_norm_[ids];
    return Xexp, yrem_exp_

def get_graph_data(experiment,  X_ = None, eid_oh_ = None, yrem_norm_ = None, 
                   n_sampled_graphs = 100, nnodes = 3, min_spacing = 20,
                   nseq_range = 100, fixed_spacing_indices = False, node_time_scaling = 5.):
     
    # For computational efficiency the number of nodes and edges in each graph is the same. 
    # For efficiency in creating the dataset, the nodes and edges are also created in parallel.
    exp_dat = data_from_experiment(experiment, X_ = X_, eid_oh_ = eid_oh_, yrem_norm_ = yrem_norm_)
    ntotal_inds = exp_dat[0].shape[0];
    inds = get_indices(n_sampled_graphs, ntotal_inds, nseq_range = nseq_range,
                          npoints_per_seq = nnodes, nmin_idx_diff = min_spacing, 
                          fixed_spacing_indices = fixed_spacing_indices)
    X__, y__ = exp_dat;
    node_attr  = [X__[inds_,...] for inds_ in inds.T];
    node_times = [y__[inds_]*node_time_scaling for inds_ in inds.T]; # to be used for making attributes for the edges.
    nodes= [Node(node_attr_) for node_attr_ in node_attr];
    
    ## Connect all edges with all previous edges:
    edges = []
    for i in range(len(nodes)):
        node_to_idx = i
        
        if node_to_idx == 0:
            next #first node does not have an incoming node.
            
        for node_from_idx in range(0, node_to_idx):
            y_from, y_to = [node_times[ni] for ni in [node_from_idx, node_to_idx]]
            edge_attr = y_to - y_from
            #print("node_from/to: %i %i"%(node_from_idx, node_to_idx))
            edges.append(Edge(edge_attr[:,np.newaxis], node_from = nodes[node_from_idx], node_to = nodes[node_to_idx]));
    g__ =Graph(nodes,edges)
    g__.node_times = node_times
    g__.inds = inds
    return g__,node_times[-1] #Returns a graph and a prediction for the time at the graph's destination node.
    
    

def get_graph_data_multiple_experiments(experiments,X_ = None, eid_oh_ = None, yrem_norm_ = None,
                                        nsamples_per_experiment = None,nnodes = None, min_spacing = None, 
                                        nseq_range = None,fixed_spacing_indices = False):
    all_graph_data = []
    for e in experiments:
        g = get_graph_data(e,  X_ = X_, eid_oh_ = eid_oh_,
                           yrem_norm_ = yrem_norm_,n_sampled_graphs = nsamples_per_experiment,
                           nnodes = nnodes, min_spacing = min_spacing, nseq_range = nseq_range,
                           fixed_spacing_indices = fixed_spacing_indices)
        all_graph_data.append(g)
    return all_graph_data

def get_multi_batch(nsamples_per_experiment, dataset_object, source_ds = True, nseq_range = None,
                    nnodes = None, min_spacing = None,fixed_spacing_indices = False):
    # In order to keep the datapoints from each experiment ballanced I'm
    # sampling the same number of graphs from each experiment

    if source_ds:
        args = dataset_object.inds_exp_source
    else:
        args = dataset_object.inds_exp_target

    kwargs = {"X_" : dataset_object.X,
              "yrem_norm_" : dataset_object.yrem_norm,
             "eid_oh_" : dataset_object.eid_oh,
             "nsamples_per_experiment" : nsamples_per_experiment,
             "nnodes" : nnodes,
              "nseq_range" : nseq_range,
              "min_spacing" : min_spacing,
             "fixed_spacing_indices" : fixed_spacing_indices}

    return get_graph_data_multiple_experiments(args, **kwargs)


if __name__ == "__main__":
    # Usage examples: (These should not fail - make a test at some point)
    i = get_indices(100, 100, nmin_idx = 10,nseq_range = 20, nmin_idx_diff = 2, npoints_per_seq=19, fixed_spacing_indices=True)

    # Get graph data:
    import numpy 
    batch = 100;
    d1 = 10;
    X = np.random.randn(batch, d1,d1), 
    eid_oh = np.random.randn(batch,1)
    yrem_norm = np.random.randn(batch,1)
    g1= get_graph_data(6, X_ = X, eid_oh_=eid_oh, yrem_norm_= yrem_norm,
               n_sampled_graphs = 100, nnodes = 5,min_spacing = 20,
               nseq_range = 500, fixed_spacing_indices = False, node_time_scaling = 1.)


    n = get_graph_data_multiple_experiments(inds_exp_source,X_ = X, 
                                        eid_oh_= eid_oh, 
                                        yrem_norm_ = yrem_norm, 
                                        nsamples_per_experiment= 100, nnodes = 5,
                                        min_spacing = 2,nseq_range=10)
