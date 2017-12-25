import unittest
import network
import numpy as np
import calculator as calc


class TestNetwork(unittest.TestCase):
    """
    test methods for network
    """

    def test_load_data(self):
        X, Y = network.load_data("data/unittest/X.txt", "data/unittest/Y.txt")
        self.assertEqual((64, 4), X.shape)
        self.assertEqual((3, 8), Y.shape)

    def test_initialize_parameters(self):
        params = network.initialize_parameters([3, 5, 6, 1])
        self.assertEqual(params["W1"].shape, (5, 3))
        self.assertEqual(params["b1"].shape, (5, 1))
        self.assertEqual(params["W2"].shape, (6, 5))
        self.assertEqual(params["b2"].shape, (6, 1))
        self.assertEqual(params["W3"].shape, (1, 6))
        self.assertEqual(params["b3"].shape, (1, 1))

        # テスト ランダムパラメータ * 0.01のテスト
        parameters = network.initialize_parameters([5, 4, 3], 3)

        W1 = np.array([(0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388),
                       (-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218),
                       (-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034),
                       (-0.00404677, -0.0054536,  -0.01546477,  0.00982367, -0.01101068)])
        assert np.allclose(parameters["W1"], W1)

        W2 = np.array([(-0.01185047, -0.0020565, 0.01486148, 0.00236716),
                       (-0.01023785, -0.00712993, 0.00625245, -0.00160513),
                       (-0.00768836, -0.00230031, 0.00745056, 0.01976111)])
        assert np.allclose(parameters["W2"], W2)

    def test_linear_forward(self):
        W = np.array([(1, 2, 3), (4, 5, 6)])
        A_prev = np.array([(1), (2), (3)])
        b = 1
        ans = np.array([15, 33])
        Z, cache = network.linear_forward(A_prev, W, b)
        assert np.allclose(Z, ans)

    def test_linear_activation_forward(self):
        W = np.array([(1, 1), (1, 2)])
        A_prev = np.array([(1), (1)])
        b = -2
        ans = np.array([(0.5), (0.73105858)])
        A, cache = network.linear_activation_forward(A_prev, W, b, "sigmoid")
        assert np.allclose(A, ans)

    def test_forward_propagation(self):
        X = np.random.randn(5, 1)
        activations = ["no use", "relu", "relu", "relu", "relu", "sigmoid"]
        layer_dims = [5, 10, 15, 10, 3, 3]
        parameters = network.initialize_parameters(layer_dims)
        AL, cache = network.forward_propagation(X, parameters, activations)
        self.assertEqual((3, 1), AL.shape)

    def test_compute_cost(self):
        AL = np.array([(0.999999, 0.999999), (0.2, 0.2), (0.3, 0.3)])
        Y = np.array([(1, 1), (0.3, 0.3), (0.1, 0.1)])
        cost = network.compute_cost(AL, Y)
        self.assertEqual(cost, 1.0804375896281297)


if __name__ == "__main__":
    unittest.main()
