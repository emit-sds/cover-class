import unittest
import numpy as np
from torch import FloatTensor
import torch
from sklearn.decomposition import PCA

from cover_class.subsample import subsampler # type: ignore[import]

RANDOM_SEED = 42

class subsampleTest(unittest.TestCase):
    def preamble(self):
        np.random.seed(RANDOM_SEED)
        self.N_datapoints = 1_000
        self.data_matrix = np.random.random(size=(self.N_datapoints, 500))
        self.num_pc = 3

    def test_convex_hull(self):
        self.preamble()
        n_samples = 3
        sample = subsampler.convex_hull(self.data_matrix, self.num_pc, n_samples)
        self.assertIsInstance(sample, FloatTensor)
        self.assertEqual(sample.dtype, torch.float32)
        self.assertEqual(len(sample), n_samples)
        self.assertEqual(sample.shape[1], self.data_matrix.shape[1])

        sample = subsampler.convex_hull(self.data_matrix, self.num_pc, self.N_datapoints+1)
        self.assertLess(len(sample), self.N_datapoints)

        manual_convex_hull_check = np.array(
            [[0, 0, 0],
             [1, 1, 1],
             [2, 2, 2],
             [1, 2, 1],
             [2, 1, 2],
             [3, 0, 0],
             [0, 3, 0],
             [0, 0, 3]]
        )
        sample = subsampler.convex_hull(manual_convex_hull_check, 2, len(manual_convex_hull_check))
        pca = PCA(
            n_components=2, svd_solver="arpack", random_state=0
        ).fit_transform(manual_convex_hull_check)
        vertices = np.unique([int(pca[:,0].argmin()), int(pca[:,0].argmax()), int(pca[:,1].argmin()), int(pca[:,1].argmax())])
        vertices = torch.from_numpy(manual_convex_hull_check[vertices]).to(dtype=torch.float32)
        self.assertTrue(all([i in sample for i in vertices]))


    def test_kmeans(self):
        '''This just tests that the function itself works'''
        self.preamble()
        n_samples = 10
        sample = subsampler.kmeans(self.data_matrix, self.num_pc, n_samples)
        self.assertIsInstance(sample, FloatTensor)
        self.assertEqual(sample.dtype, torch.float32)
        self.assertEqual(len(sample), n_samples)
        self.assertEqual(sample.shape[1], self.data_matrix.shape[1])

    def test_lhs(self):
        '''This just tests that the function itself works'''
        self.preamble()
        hypercubes_per_dimension = 3
        samples_per_hypercube = 5

        sample = subsampler.lhs(self.data_matrix, self.num_pc, hypercubes_per_dimension, samples_per_hypercube)
        self.assertIsInstance(sample, FloatTensor)
        self.assertEqual(sample.dtype, torch.float32)
        self.assertLess(len(sample), ((self.num_pc ** hypercubes_per_dimension)*samples_per_hypercube)+1)
        self.assertEqual(sample.shape[1], self.data_matrix.shape[1])

if __name__ == "__main__":
    unittest.main()