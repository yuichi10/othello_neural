import unittest
import calculator
import numpy as np


class TestNetwork(unittest.TestCase):
    """
    test methods of calculator
    """

    def test_sigmoid(self):
        expect = calculator.sigmoid(np.array([0, 2]))
        self.assertEqual(0.5, expect[0])
        self.assertEqual(0.88079707797788231, expect[1])

    def test_sigmoid_derivative(self):
        expect = calculator.sigmoid_derivative(np.array([1, 2, 3]))
        self.assertEqual(0.19661193324148185, expect[0])
        self.assertEqual(0.10499358540350662, expect[1])
        self.assertEqual(0.045176659730911999, expect[2])

    def test_relu(self):
        expect = calculator.relu(np.array([(1, 2), (-1, 0)]))
        ans = np.array([(1, 2), (0, 0)])
        assert np.allclose(expect, ans)

    def test_relu_derivative(self):
        expect = calculator.relu_derivative(np.array([(1, 2), (-1, 0)]))
        ans = np.array([(1, 1), (0, 0)])
        assert np.allclose(expect, ans)


if __name__ == "__main__":
    unittest.main()
