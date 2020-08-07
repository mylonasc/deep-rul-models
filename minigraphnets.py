""" Classes for basic manipulation of GraphNet """
import numpy as np
import tensorflow as tf


class Node:
    def __init__(self, node_attr_tensor):
        if len(node_attr_tensor.shape) <2:
            raise ValueError("The shape of the input for nodes and edges should have at least 2 dimensions!")
        self.node_attr_tensor = node_attr_tensor
        self.incoming_edges = [];
        self.shape = self.node_attr_tensor.shape
        
    def get_state(self):
        return self.node_attr_tensor
    
    def set_tensor(self, tensor):
        self.node_attr_tensor = tensor
        self.shape = self.shape = tensor.shape
        
    def copy(self):
        if isinstance(self.node_attr_tensor , np.ndarray):
            node_attr_tensor = self.node_attr_tensor.copy()
        else:
            node_attr_tensor = self.node_attr_tensor # if this is an eager tensor, the assignment copies the tensor.
        return Node(node_attr_tensor)

    def __add__(self, n):
        return Node(self.node_attr_tensor + n.node_attr_tensor)

    def __sub__(self, n):
        return Node(self.node_attr_tensor  - n.node_attr_tensor)
    
# My implementation relies on eager mode and all computation happens in place. In reality only nodes
# and edges have data and the graph class is just for defining the computation between them.
class Edge:
    def __init__(self, edge_attr_tensor, node_from, node_to):
        self.edge_tensor = edge_attr_tensor
        self.node_from = node_from
        self.node_to = node_to
        self.shape = self.edge_tensor.shape
        
        # Keep a reference to this edge since it is needed for aggregation afterwards.
        node_to.incoming_edges.append(self)

    def set_tensor(self, edge_tensor):
        self.edge_tensor = edge_tensor
        self.shape = edge_tensor.shape
    
    def copy(self, nodes_correspondence):
        if isinstance(self.edge_tensor, np.ndarray):
            edge_tensor = self.edge_tensor.copy()
        else:
            edge_tensor = self.edge_tensor # If this is an eager tensor, this a copy

        node_from = nodes_correspondence[self.node_from]
        node_to   = nodes_correspondence[self.node_to]
        return Edge(edge_tensor, node_from, node_to)

    def __add__(self, edge):
        print("Edge addition is not implemented!")
        assert(0)
        


class Graph:
    def __init__(self, nodes, edges, NO_VALIDATION=True):
        self.nodes = nodes
        self.edges = edges
        if not NO_VALIDATION:
            self.validate_graph()


    def is_equal_by_value(self,g2):
        """
        Checks if the graphs have the same values for node and edge attributes
        """
        is_equal = True
        for n1,n2 in zip(self.nodes, g2.nodes):
            is_equal = is_equal and tf.reduce_all(n1.node_attr_tensor == n2.node_attr_tensor)

        for e1, e2 in zip(self.edges, g2.edges):
            is_equal = is_equal and tf.reduce_all(e1.edge_tensor== e2.edge_tensor)
        
        return bool(is_equal)
    
    def compare_connectivity(self,g2):
        """
        Checks if the connectivity of two graphs is the same.
        """
        g1 = self
        nodes_from_match = [(g1.nodes.index(e1.node_from) == g2.nodes.index(e2.node_from)) for e1,e2 in zip(g1.edges,g2.edges)]
        nodes_to_match = [(g1.nodes.index(e1.node_to) == g2.nodes.index(e2.node_to)) for e1,e2 in zip(g1.edges,g2.edges)]
        all_matching = True
        for matches in [*nodes_from_match, *nodes_to_match]:
            all_matching = all_matching and matches
        return all_matching



    @staticmethod
    def validate_graph(self):

        # validate that the edges are all 
        for e in self.edges:

            if ((e.node_from in self.nodes)):
                raise AssertionError("The source node {nn} for edge {ee} is not in the graph!".format(nn = e.node_from, ee = e))
            if (e.node_to in self.nodes):
                raise AssertionError("The destination node {nn} for edge {ee} is not in the graph!".format(nn = e.node_to, ee = e))


    def copy(self):
        # copy attributes of nodes and edges and re-create graph connectivity:
        nodes_coppied = [n.copy() for n in self.nodes]
        nodes_correspondence = {s:c for s , c in zip(self.nodes,nodes_coppied)}
        # Instantiate the new edges:
        coppied_edge_instances = []
        for e in self.edges:
            #if isinstance(e.edge_tensor, np.ndarray):
            #    edge_val = e.edge_tensor.copy()
            #else:
            #    edge_val = e.edge_tensor
            enew = e.copy(nodes_correspondence) #Edge(edge_val, nodes_correspondence[e.node_from], nodes_correspondence[e.node_to])
            coppied_edge_instances.append(enew)
        return Graph(nodes_coppied, coppied_edge_instances)

    def get_subgraph_from_nodes(self, nodes):
        """
        Node should belong to graph. Creates a new graph with coppied edge and
        node properties, defined from a sub-graph of the original graph.
        parameters:
          self (type = Graph): the graph we want a sub-graph from
          nodes: the nodes of the graph we want the subgraph of.
        """
        sg_nodes_copy = [n.copy() for n in nodes]
        original_copy_nodes_correspondence = {n:nc for n, nc in zip(nodes, sg_nodes_copy)}
        sg_edges_copy = [];
        if len(self.edges) > 0:
            for e in self.edges:
                if (e.node_from in nodes) and (e.node_to in nodes):
                    sg_edges_copy.append(e.copy(original_copy_nodes_correspondence))

        g = Graph(sg_nodes_copy, sg_edges_copy)
        return g

    def __add__(self, graph):
        """
        This should only work with graphs that have compatible node and edge features
        Assumed also that the two graphs have the same connectivity (otherwise this will fail ugly)
        """
        nodes = [nself + n for nself,n in zip(self.nodes,graph.nodes)]
        correspondence = {s:t for s, t in zip(self.nodes,nodes)}
        added_edges = [];
        for eself,e in zip(self.edges, graph.edges):
            enew = Edge(eself.edge_tensor +  e.edge_tensor, 
                    correspondence[eself.node_from], 
                    correspondence[eself.node_to])
            added_edges.append(enew);

        return Graph(nodes, added_edges)



def make_graph_tuple_from_graph_list(list_of_graphs):
    """
    Takes in a list of graphs (with consistent sizes - not checked)
    and creates a graph tuple (input tensors + some book keeping)
    
    Because there is some initial functionality I don't want to throw away currently, that implements special treatment for nodes and edges
    coming from graphs with the same topology, it is currently required that the first dimension of nodes and edges
    for the list of graphs that are entered in this function is always 1 (this dimension is the batch dimension in the previous implementation.)
    """
    # graph_id = [id_ for id_, dummy in enumerate(list_of_graphs)]
    all_edges, all_nodes, n_nodes,n_edges =[[],[],[],[]]
    for g in list_of_graphs:
        all_edges.extend(g.edges)
        all_nodes.extend(g.nodes)
        n_nodes.append(len(g.nodes)) 
        n_edges.append(len(g.edges)) 
    
    edge_attr_tensor, nodes_attr_tensor, senders, receivers = [[],[],[],[]];
    for e in all_edges:
        edge_attr_tensor.append(e.edge_tensor)
        senders.append(all_nodes.index(e.node_from))
        receivers.append(all_nodes.index(e.node_to))
        
        #senders.append(e.node_from.find(gin.nodes))
        #receivers.append(e.node_to.find(gin.nodes))
    
    for n in all_nodes:
        nodes_attr_tensor.append(n.node_attr_tensor)
    
    edges_attr_stacked = tf.stack(edge_attr_tensor,0)
    nodes_attr_stacked = tf.stack(nodes_attr_tensor,0)
    return GraphTuple(nodes_attr_stacked, edges_attr_stacked,senders, receivers, n_nodes, n_edges)# , graph_id)


class GraphTuple:
    def __init__(self, nodes, edges,senders,receivers, n_nodes, n_edges, sort_receivers_to_edges  = False):
        """
        A graph tuple contains multiple graphs for faster batched computation. 
        
        parameters:
            nodes      : a `tf.Tensor` containing all the node attributes
            edges      : a `tf.Tensor` containing all the edge attributes
            senders    : a list of sender node indices defining the graph connectivity. The indices are unique accross graphs
            receivers  : a list of receiver node indices defining the graph connectivity. The indices are unique accross graphs
            n_nodes    : a list, a numpy array or a tf.Tensor containing how many nodes are in each graph represented by the nodes and edges in the object
            n_edges    : a list,a numpy array or a tf.Tensor containing how many edges are in each graph represented by the nodes and edges in the object
            sort_receivers :  whether to sort the edges on construction, allowing for not needing to sort the output of the node receiver aggregators.
        """
        # Sort edges according to receivers and sort receivers:
        assert(len(n_nodes) == len(n_edges))
        
        self.nodes = nodes # floats tensor
        self.edges = edges # floats tensor
        self.senders = senders     # integers
        self.receivers = receivers # integers
        self.n_nodes = n_nodes     # integers
        self.n_edges = n_edges     # integers
        self.n_graphs = len(self.n_nodes)
        
    def get_graph(self, graph_index):
        """
        Returns a new graph with the same properties as the original  graph.
        gradients are not traced through this operation.
        """
        assert(graph_index >=0 )
        if graph_index > self.n_graphs:
            raise ValueError("The provided index is larger than the available graphs in this GraphTuple object.")
            
        get_start_stop_index = lambda sizes_list, index : np.cumsum([0,*sizes_list[0:index+1]])[-2:]
        start_idx_nodes , end_idx_nodes = get_start_stop_index(self.n_nodes, graph_index)
        start_idx_edges , end_idx_edges = get_start_stop_index(self.n_edges, graph_index)
        nodes_attrs = self.nodes[start_idx_nodes:end_idx_nodes]
        senders, receivers, edge_attr = [v[start_idx_edges:end_idx_edges] for v in [self.senders, self.receivers,self.edges]]
        senders = senders-start_idx_nodes
        receivers = receivers - start_idx_nodes
        nodes = [Node(node_attr) for node_attr in nodes_attrs]
        edges = [Edge(edge_attr_tensor, nodes[node_from_idx], nodes[node_to_idx]) for edge_attr_tensor, node_from_idx, node_to_idx in zip(edge_attr, senders,receivers)]
        return Graph(nodes, edges)

        

