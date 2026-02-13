from typing import Tuple
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader

from cover_class.dataloader import OrchestratorDataset, OrchestratorDatasetArgs # type: ignore[import]
from cover_class.simulation import SimulationArgs, DataArgs # type: ignore[import]

RANDOM_SEED = 42

def make_odsa(bsz, percent, s, d, sd, sl) -> OrchestratorDatasetArgs:
    return OrchestratorDatasetArgs(
        batch_size = bsz,
        percent_static = percent,
        sim_config_args = s,
        sim_data_args = d,
        static_data = sd,
        static_labels = sl,
    )

def new_sim_args() -> Tuple[SimulationArgs, DataArgs]:
    n_classes = 10
    return (
    SimulationArgs(
        n_iters = 100,
        n_classes_in_subsets = 5,
        n_classes = 10,
        n_components = [list(range(10)) for _ in range(n_classes)],
        min_frac = 0.,
        alpha = None,
        alpha_uniform_low = 0.,
        alpha_uniform_high = 0.,
        white_noise = 0.,
        noise_scalar=None,
        noise_covariance = None,
        return_fractions = False,
        glint_scalar_range = [None, None],
        water_classes=[],
    ),
    None)
    # DataArgs(torch.tensor([.1, .2, .3]), torch.tensor([.1, .2, .3])))

class dataloaderTest(unittest.TestCase):

    def test_OrchestratorDatasetArgs(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        labels = torch.randint(0, 3, (10,), dtype=torch.long)
        sim_args, data_args = new_sim_args()

        with self.subTest("static_only_forces_100_percent_check"):
            args = make_odsa(10, 0., sim_args, data_args, data, labels)
            self.assertTrue(args._using_static)
            self.assertFalse(args._using_sim)
            self.assertEqual(args.percent_static, 1.00)
            self.assertTrue(torch.all(args._method_selection_idxs == 1))

        with self.subTest("simulation_only_forces_0_percent_check"):
            args = make_odsa(10, 1.00, sim_args, object(), None, None)
            self.assertFalse(args._using_static)
            self.assertTrue(args._using_sim)
            self.assertEqual(args.percent_static, 0.0)
            self.assertTrue(torch.all(args._method_selection_idxs == 0))

        with self.subTest("valid_simulation_and_static_data_check"):
            percent = .337
            args = make_odsa(10, percent, sim_args, object(), data, labels)
            self.assertTrue(args._using_static)
            self.assertTrue(args._using_sim)
            self.assertEqual(args.percent_static, percent)
            self.assertEqual(int(args._method_selection_idxs.sum()), int(percent*100))

        with self.subTest("percent_boundaries_check"):
            for percent in (0.0, 1.00):
                args = make_odsa(10, percent, sim_args, object(), data, labels)
                self.assertEqual(args.percent_static, percent)
                self.assertEqual(int(args._method_selection_idxs.sum()), int(percent*100))

        with self.subTest("assertion_clauses_check"):
            self.assertRaises(AssertionError, make_odsa, 10, 0., None, None, None, None)
            self.assertRaises(AssertionError, make_odsa, 10, -.1, sim_args, object(), object(), object())
            self.assertRaises(AssertionError, make_odsa, 10, 1.01, sim_args, object(), object(), object())


    def test_OrchestratorDataset_simulated_only(self):
        bsz = 10
        dims = 5

        torch.manual_seed(RANDOM_SEED)
        data = torch.ones((bsz, dims))
        labels = torch.zeros((bsz,dims), dtype=torch.int32)
        sim_args, data_args = new_sim_args()
        def mock_run_simulation(_cfg, _data): return data, labels, None

        args = make_odsa(bsz, 0.0, sim_args, object(), None, None)

        with patch("cover_class.simulation.run_simulation", side_effect=mock_run_simulation):
            ds = OrchestratorDataset(args)
            dl = DataLoader(ds, batch_size=None)

            i = 0
            for X, Y in dl:
                self.assertTrue(ds.is_simulated_batch)
                self.assertIsInstance(X, torch.FloatTensor)
                self.assertEqual(X.shape, (bsz, dims))
                self.assertTrue(torch.allclose(X, data))
                self.assertTrue(torch.all(Y[:, 0] == 1)) # testing the labels this way since they're one-hot
                self.assertTrue(torch.all(Y[:, 1:] == 0))
                self.assertEqual(ds.static_epoch, 0)
                self.assertEqual(ds.static_epoch_step, 0)
                self.assertEqual(ds.step, i+1)
                
                i += 1
                if i == bsz*10: break

    def test_OrchestratorDataset_static_only(self):
        torch.manual_seed(RANDOM_SEED)

        N, bsz, dims, epochs = 40, 10, 3, 5

        data = torch.arange(N * dims, dtype=torch.float32).reshape(N, dims)
        labels = torch.arange(N, dtype=torch.long)

        # sim_args, data_args = new_sim_args()
        args = make_odsa(bsz, 100.0, None, None, data, labels)
        ods = OrchestratorDataset(args, shuffle=True)
        dl = DataLoader(ods, batch_size=None)

        seen_rows = []
        seen_labels = []

        i, t = 0, 0
        for X, Y in dl:
            i += 1; t += 1
            if i == (N//bsz): i = 0
            if t == (N//bsz)*epochs: break
            self.assertFalse(ods.is_simulated_batch)
            self.assertEqual(X.shape, (bsz, dims))
            self.assertEqual(Y.shape, (bsz, N))
            self.assertEqual(ods.static_epoch_step, i)
            self.assertEqual(ods.step, t)
            if t <= (N//bsz):
                seen_rows.append(X)
                seen_labels.append(Y)
        self.assertEqual(ods.static_epoch, epochs)

        X_all = torch.vstack(seen_rows)
        y_all = torch.vstack(seen_labels)

        self.assertEqual(y_all.sum(), N) # test the sum for the one-hot encoding
        self.assertTrue(torch.equal(y_all.sum(1), torch.ones(N)))

    def test_OrchestratorDataset_mixed_static_and_sim(self):
        bsz, dims = 10, 5
        percent_static = 0.60
        sim_args, _ = new_sim_args()

        torch.manual_seed(RANDOM_SEED)
        static_data = torch.ones((bsz, dims))
        static_labels = torch.ones(bsz, dtype=torch.long)
        sim_data = static_data + 3
        sim_labels = torch.ones(bsz, sim_args.n_classes, dtype=torch.long) + 2
        def mock_run_simulation(_cfg, _data): return sim_data, sim_labels, None
        def mock_reset(): ...

        args = make_odsa(bsz, percent_static, sim_args, object(), static_data, static_labels)

        with patch("cover_class.simulation.run_simulation", side_effect=mock_run_simulation), \
            patch("cover_class.dataloader.OrchestratorDataset.__reset__", side_effect=mock_reset):
            ods = OrchestratorDataset(args, shuffle=True)
            dl = DataLoader(ods, batch_size=None)

            static_count = 0
            sim_count = 0
            i = 0
            for X, Y in dl:
                if ods.is_simulated_batch:
                    sim_count += 1
                    self.assertTrue(torch.allclose(X, sim_data))
                    self.assertTrue(torch.equal(Y[:, sim_labels[0, 0]], torch.ones(len(Y))))
                else:
                    static_count += 1
                    dl.dataset.static_epoch_step = 0
                    self.assertTrue(torch.allclose(X, static_data))
                    self.assertTrue(torch.equal(Y[:, 1], static_labels))
                i += 1
                if i == 100: break

            self.assertEqual(static_count, int(percent_static*100))
            self.assertEqual(sim_count, 100 - int(percent_static*100))

if __name__ == "__main__":
    unittest.main()