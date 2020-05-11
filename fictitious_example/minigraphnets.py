""" Classes for basic manipulation of GraphNet """

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
        node_attr_tensor = self.node_attr_tensor
    
    
    
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
        edge_tensor = self.edge_tensor
        node_from = nodes_correspondence[self.node_from]
        node_to   = nodes_correspondence[self.node_to]
        return Edge(edge_tensor, node_from, node_to)
        
class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        
    def copy(self):
        # copy attributes of nodes and edges and re-create graph connectivity:
        nodes_coppied = [Node(n.node_attr_tensor.copy()) for n in self.nodes]
        nodes_correspondence = {s:c for s , c in zip(self.nodes,nodes_coppied)}
        # Instantiate the new edges:
        coppied_edge_instances = []
        for e in self.edges:
            enew = Edge(e.edge_tensor, nodes_correspondence[e.node_from], nodes_correspondence[e.node_to])
            coppied_edge_instances.append(enew)
        return Graph(nodes_coppied, coppied_edge_instances)
        
