import unittest 

class TestGraphDatastructures(unittest.TestCase):

    def test_construct_nodes_edges_simple_graph_np(self):
        """
        Tests the construction of some basic datastructures useful for GraphNet computation
        """
        n1 = Node(np.random.randn(10,10))
        n2 = Node(np.random.randn(10,10))
        e12 = Edge(np.random.randn(5,10),n1,n2)
        g = Graph([n1,n2], [e12])

        
    def test_node_operations(self):

        r1 = np.random.randn(10,10)
        r2 = np.random.randn(10,10)
        n1 = Node(r1)
        n2 = Node(r2)
        n3 = n1  + n2
        self.assertEqual(np.linalg.norm(n2.node_attr_tensor + n1.node_attr_tensor-n3.node_attr_tensor),0)

    def test_node_copy(self):
        """
        test that when copying the object the value is coppied but not the 
        reference
        """
        n1 = Node(np.random.randn(10,10))
        n2 = n1.copy()
        self.assertTrue(n1 != n2)
        self.assertTrue(np.linalg.norm((n1 - n2).node_attr_tensor)== 0.)

    def test_graph_tuple_construction(self):
        """
        Tests if I can properly set and then retrieve a graph tuple.
        """
        batch_size = 1
        node_input_size = 2
        edge_input_size = 2
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        n4 = Node(np.random.randn(batch_size, node_input_size))
        n5 = Node(np.random.randn(batch_size, node_input_size))

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        e34 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n3, node_to = n4)
        e45 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n4, node_to = n5)

        g1 = Graph([n1,n2],[e12])
        g2 = Graph([n1,n2,n3,n4],[e12,e21,e23,e34])
        g3 = Graph([n3, n4] , [e34])

        from minigraphnets import GraphTuple, make_graph_tuple_from_graph_list
        old_graphs_list = [g1.copy(),g2.copy(),g3.copy()]
        graph_tuple = make_graph_tuple_from_graph_list(old_graphs_list)
        new_graphs_list = [graph_tuple.get_graph(k) for k in range(graph_tuple.n_graphs)]
        self.assertTrue(np.all([(k.is_equal_by_value(m) and k.compare_connectivity(m) ) for k, m in zip(new_graphs_list, old_graphs_list)]))


class TestGraphNet(unittest.TestCase):
    def test_construct_simple_eval_graphnet(self):
        from graphnet_utils import GraphNet, GraphNetFunctionFactory
        edge_input_size = 15
        node_input_size = 10
        node_output_size, edge_output_size = node_input_size, edge_input_size
        node_input = tf.keras.layers.Input(shape = (node_input_size,))
        edge_input = tf.keras.layers.Input(shape = (edge_input_size,))

        node_function = tf.keras.Model(outputs = tf.keras.layers.Dense(node_output_size)(node_input), inputs= node_input)
        edge_function = tf.keras.Model(outputs = tf.keras.layers.Dense(edge_output_size)(edge_input), inputs= edge_input)
        edge_aggregation_function = GraphNetFunctionFactory.make_edge_aggregation_function(edge_output_size)
        graphnet = GraphNet(node_function = node_function, edge_function = edge_function, edge_aggregation_function = edge_aggregation_function, node_to_prob = None)
        batch_size = 10
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        g = Graph([n1,n2],[e12])

    def test_eval_modes(self):
        """
        test the different evaluation modes.
        There are 3 evaluation modes - one appropriate for batched graphs, and two for graphs of the same shape ("batched" or unbached ("safe")).
        The "safe" mode is used as reference for the correct results; All modes should give the same output within an error margin (due to finite precission 
        rounding errors and the different comp. graphs.)
        """
        from graphnet_utils import GraphNet, make_mlp_graphnet_functions

        batch_size = 12
        tf.keras.backend.set_floatx("float64")
        node_input_size = 10
        edge_input_size = node_input_size

        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        node_abs_vals = [np.abs(n.node_attr_tensor) for n in [n1,n2,n3]]

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        edge_abs_vals = [np.abs(e.edge_tensor) for e in [e12,e21,e23]]

        g1 = Graph([n1,n2,n3],[e12,e21,e23])

        ## The non-graph independent version:
        gi = False
        graph_fcn = make_mlp_graphnet_functions(150, input_size = node_input_size, output_size = node_input_size, graph_indep=gi)
        graph_fcn.update({"graph_independent" : gi})
        gn = GraphNet(**graph_fcn )
        res1 = gn.graph_eval(g1.copy(),eval_mode = "safe")
        res2 = gn.graph_eval(g1.copy(), eval_mode = "batched")
        error_nodes = np.max([np.linalg.norm(n1.node_attr_tensor - n2.node_attr_tensor) for n1, n2 in zip(res1.nodes, res2.nodes)])/np.min(node_abs_vals)
        error_edges = np.max([np.linalg.norm(e1.edge_tensor - e2.edge_tensor) for e1,e2 in zip(res1.edges, res2.edges)])/np.min(edge_abs_vals)
        #print(error_nodes, error_edges)
        self.assertTrue(error_nodes < 1e-10)
        self.assertTrue(error_edges < 1e-10)

        ## The graph-independent version:
        gi = True
        graph_fcn = make_mlp_graphnet_functions(150, input_size = node_input_size, output_size = node_input_size, graph_indep=gi)
        graph_fcn.update({"graph_independent" : gi})
        gn = GraphNet(**graph_fcn )
        res1 = gn.graph_eval(g1.copy(),eval_mode = "safe")
        res2 = gn.graph_eval(g1.copy(), eval_mode = "batched")
        error_nodes = np.max([np.linalg.norm(n1.node_attr_tensor - n2.node_attr_tensor) for n1, n2 in zip(res1.nodes, res2.nodes)])/np.min(node_abs_vals)
        error_edges = np.max([np.linalg.norm(e1.edge_tensor - e2.edge_tensor) for e1,e2 in zip(res1.edges, res2.edges)])/np.min(edge_abs_vals)
        #print(error_nodes, error_edges)
        self.assertTrue(error_nodes < 1e-10)
        self.assertTrue(error_edges < 1e-10)


    def test_graph_tuple_eval(self):
        """
        The graph tuples are graphs of different sizes batched to a single object,
        to allow for more single-instruction multiple-data computation (batched computation).
        This is the only evalution mode deep mind's graphnets implement directly.
        """
        None

        

        



if __name__ == "__main__":

    from minigraphnets import Node, Edge, Graph
    import tensorflow as tf
    import numpy as np
    unittest.main(verbosity = 2)

