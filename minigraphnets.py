""" Classes for basic manipulation of GraphNet """
import numpy as np

class Node:
    def __init__(self, node_attr_tensor):
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
            edge_tensor = self.edge_tensor.numpy().copy()
        else:
            edge_tensor = self.edge_tensor # If this is an eager tensor, this a copy

        node_from = nodes_correspondence[self.node_from]
        node_to   = nodes_correspondence[self.node_to]
        return Edge(edge_tensor, node_from, node_to)

    def __add__(self, edge):
        print("Edge addition is not implemented!")
        assert(0)
        

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        
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
