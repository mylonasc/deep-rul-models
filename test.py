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

    def test_correct_sizes_graphnet_mlp(self):
        """
        Test sizes of intermediate computations when using a default MLP construction.
        """
        from graphnet_utils import GraphNet
        



if __name__ == "__main__":

    from minigraphnets import Node, Edge, Graph
    import tensorflow as tf
    import numpy as np
    unittest.main(verbosity = 2)

