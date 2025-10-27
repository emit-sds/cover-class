import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

from cover_class.subsample.forward_pipeline import subsample_from_config # type: ignore[import]
from cover_class.subsample import forward_pipeline # type: ignore[import]

class subsampleTest(unittest.TestCase):
    @patch('cover_class.subsample.forward_pipeline.convex_hull')
    @patch('cover_class.subsample.forward_pipeline.kmeans')
    @patch('cover_class.subsample.forward_pipeline.kmedoids')
    @patch('cover_class.subsample.forward_pipeline.lhs')
    @patch('cover_class.subsample.forward_pipeline.read_config')
    def test_subsample_from_config(self, rc_mock:Mock, lhs_mock:Mock, kmed_mock:Mock, km_mock:Mock, cv_mock:Mock):
        config = {
            'subsample': {
                'selected-method':'',
                'convex-hull': {'num_pc':1, 'n_samples':2, 'qhull_options': 'x'},
                'kmeans': {'num_pc':1, 'n_samples':2, 'init': 'x'},
                'kmedoids': {'num_pc':1, 'n_samples':2, 'metric': 'euclidean'},
                'lhs': {'num_pc':1, 'n_samples':2, 'hypercubes_per_dimension': 3, 'samples_per_hypercube':4},
                'fail': {},
            },
        }
        data = np.array([0xD, 0xE, 0xA, 0xD, 0xD, 0xE, 0xA, 0xD])

        def check_args(item):
            t, m = item
            config['subsample']['selected-method'] = t
            rc_mock.side_effect = lambda x: config
            subsample_from_config('/path', data)
            m.assert_called_once()
            call_args = m.call_args_list.pop()
            self.assertDictEqual(config['subsample'][t], call_args.kwargs) # type: ignore 
        list(map(check_args, {'convex-hull':cv_mock, 'kmeans':km_mock, 'kmedoids':kmed_mock, 'lhs':lhs_mock}.items()))

        self.assertRaises(ValueError, check_args, ('fail', cv_mock))

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

        result = forward_pipeline.drop_bad_bands(data_matrix.numpy(), banddef.numpy(), drop_wl_ranges)

        expected = torch.tensor([[0, 3],
                                 [5, 8]], dtype=torch.float32)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, expected.shape)

if __name__ == "__main__":
    unittest.main()