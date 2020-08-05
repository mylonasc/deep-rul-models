""" Classes for basic manipulation of GraphNet """
import numpy as np

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


class GraphTuple:
    def __init__(self, graphs):
        """
        A class encapsulating computation with arbitrary graphs in one batch.
        This is faster as the edges and nodes for different graphs are evaluated in parallel.
        Each graph in the tuple takes an index and this index is used to keep the correspondence of edges and nodes to the graph.
        Computation steps:
        --------------------
        (1) computing edges:
        ---[edges graph1][edges graph2][...     ...]----
        ---[graph1 index][graph2 index][...     ...]---- <- this is used for book keeping
        ---[... total batch of edges to be computed]---- <- batch of computation

        (2) Aggregating edges:
            *
        (3) computing nodes:
        ---[nodes graph1][nodes graph2][...     ...]----
        ---[graph1 index][graph2 index][...     ...]---- <- this is used for book keeping
        ---[... total batch of nodes to be computed]---- <- batch of computation

        *aggregating edges:

        """
        self.graphs = graphs


