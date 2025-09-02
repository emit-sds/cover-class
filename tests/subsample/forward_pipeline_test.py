import unittest
from unittest.mock import Mock, patch
import numpy as np

from cover_class.subsample.forward_pipeline import subsample_from_config # type: ignore[import]

class subsampleUtilsTest(unittest.TestCase):
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

        map(check_args, {'convex-hull':cv_mock, 'kmeans':km_mock, 'kmedoids':kmed_mock, 'lhs':lhs_mock}.items())

        
if __name__ == "__main__":
    unittest.main()