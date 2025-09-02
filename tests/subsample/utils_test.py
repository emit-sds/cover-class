import unittest
import torch

from cover_class.subsample import utils # type: ignore[import]

class subsampleTest(unittest.TestCase):
    def test_train_test_split(self):
        frac = 0.2
        x = torch.rand((10,5))
        y = torch.ones(len(x))
        x_train, x_test, y_train, y_test = utils.train_test_split(x, y, frac)
        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(len(y_test), 2)
        self.assertIsInstance(x_train, torch.FloatTensor)
        self.assertIsInstance(x_test, torch.FloatTensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)

if __name__ == "__main__":
    unittest.main()