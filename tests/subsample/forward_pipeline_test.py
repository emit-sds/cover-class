import unittest
import torch

from cover_class.subsample import forward_pipeline # type: ignore[import]

class subsampleTest(unittest.TestCase):
    def test_train_test_split(self):
        frac = 0.2
        x = torch.rand((10,5))
        y = torch.ones(len(x))
        x_train, x_test, y_train, y_test = forward_pipeline.train_test_split(x, y, frac)
        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(len(y_test), 2)
        self.assertIsInstance(x_train, torch.FloatTensor)
        self.assertIsInstance(x_test, torch.FloatTensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)

    def test_drop_bad_bands(self):
        banddef = torch.tensor([400, 500, 600, 700, 800], dtype=torch.float32)
        data_matrix = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        drop_wl_ranges = [[450, 650], [800, 800]]  # Drop 500–600 nm bands and 800 nm band

        result = forward_pipeline.drop_bad_bands(data_matrix, banddef, drop_wl_ranges)

        expected = torch.tensor([[0, 3],
                                 [5, 8]], dtype=torch.float32)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, expected.shape)

if __name__ == "__main__":
    unittest.main()