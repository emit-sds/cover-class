import unittest
from unittest.mock import patch
import torch

from cover_class.dataloader import OrchestratorDataset, OrchestratorDatasetArgs # type: ignore[import]
from cover_class.simulation import SimulationArgs, DataArgs # type: ignore[import]

RANDOM_SEED = 42

class dataloaderTest(unittest.TestCase):

    def test_OrchestratorDatasetArgs(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        labels = torch.randint(0, 3, (10,), dtype=torch.long)
        
        def make_odla(bsz, percent, s, d, sd, sl) -> OrchestratorDatasetArgs:
            return OrchestratorDatasetArgs(
                batch_size = bsz,
                percent_static = percent,
                sim_config_args = s,
                sim_data_args = d,
                static_data = sd,
                static_labels = sl,
            )

        with self.subTest("static_only_forces_100_percent_check"):
            args = make_odla(10, 0., None, None, data, labels)
            self.assertTrue(args._using_static)
            self.assertFalse(args._using_sim)
            self.assertEqual(args.percent_static, 100.0)
            self.assertTrue(torch.all(args._method_selection_idxs == 1))

        with self.subTest("simulation_only_forces_0_percent_check"):
            args = make_odla(10, 100., object(), object(), None, None)
            self.assertFalse(args._using_static)
            self.assertTrue(args._using_sim)
            self.assertEqual(args.percent_static, 0.0)
            self.assertTrue(torch.all(args._method_selection_idxs == 0))

        with self.subTest("valid_simulation_and_static_data_check"):
            percent = 33.7
            args = make_odla(10, percent, object(), object(), data, labels)
            self.assertTrue(args._using_static)
            self.assertTrue(args._using_sim)
            self.assertEqual(args.percent_static, percent)
            self.assertTrue(torch.all(args._method_selection_idxs[:int(percent)] == 1))
            self.assertTrue(torch.all(args._method_selection_idxs[int(percent):] == 0))

        with self.subTest("percent_boundaries_check"):
            for percent in (0.0, 100.0):
                args = make_odla(10, percent, object(), object(), data, labels)
                self.assertEqual(args.percent_static, percent)
                self.assertEqual(int(args._method_selection_idxs.sum()), int(percent))

        with self.subTest("assertion_clauses_check"):
            self.assertRaises(AssertionError, make_odla, 10, 0., None, None, None, None)
            self.assertRaises(AssertionError, make_odla, 10, -.1, object(), object(), object(), object())
            self.assertRaises(AssertionError, make_odla, 10, 100.1, object(), object(), object(), object())


    # def test_dataloader(self):
    #     # 1. test only static data
    #     # 2. test only simulated data
    #     # 3. test both mixed together
    #     size = (10, 5)
    #     torch.manual_seed(RANDOM_SEED)
    #     static_data = torch.FloatTensor(torch.rand(size, dtype=torch.float32))
    #     static_labels = torch.rand(size, dtype=torch.int8)
    #     args = OrchestratorDatasetArgs(
    #         batch_size = 3,
    #         percent_static = 100.,
    #         static_data = static_data,
    #         static_labels = static_labels,
    #     )
    #     OrchestratorDataset()

    def test_OrchestratorDataset_simulated_only(self):
        torch.manual_seed(RANDOM_SEED)
        data = torch.ones((bsz, dims))
        labels = torch.arange(bsz)

        dummy_sim_cfg = object()
        dummy_sim_data = object()
        bsz = 10
        dims = 5

        def fake_run_simulation(_cfg, _data):
            return data, labels

        args = OrchestratorDatasetArgs(
            batch_size=bsz,
            percent_static=0.0,  # will be forced to 0 in __post_init__ anyway
            sim_config_args=dummy_sim_cfg,
            sim_data_args=dummy_sim_data,
            static_data=None,
            static_labels=None,
        )

        with patch("cover_class.dataloader.run_simulation", side_effect=fake_run_simulation):
            dl = OrchestratorDataset(args)

            for X, Y in dl:
                self.assertTrue(dl.is_simulated_batch)
                self.assertIsInstance(X, torch.FloatTensor)
                self.assertEqual(X.shape, (bsz, dims))
                self.assertTrue(torch.allclose(X, data))
                self.assertTrue(torch.equal(Y, labels))

    def test_OrchestratorDataset_static_only(self):
        """
        When only static tensors are provided, batches should be drawn from the static set,
        and an epoch reset should occur after consuming the dataset once.
        This test also detects repeated identical batches if index progression doesn’t advance.
        """
        torch.manual_seed(RANDOM_SEED)

        N = 40
        bs = 8
        feat = 3

        # Build a labeled static dataset with unique rows per index to detect repetition
        data = torch.arange(N * feat, dtype=torch.float32).reshape(N, feat)
        labels = torch.arange(N, dtype=torch.long)

        args = OrchestratorDatasetArgs(
            batch_size=bs,
            percent_static=100.0,  # will be forced to 100 in __post_init__
            sim_config_args=None,
            sim_data_args=None,
            static_data=data,
            static_labels=labels,
        )
        dl = OrchestratorDataset(args, shuffle=True)

        with self.subTest("all-batches-static-and-cover-dataset-once"):
            seen_rows = []
            seen_labels = []

            # Consume exactly N/bs batches (one epoch’s worth)
            for _ in range(N // bs):
                X, y = next(dl)
                self.assertFalse(dl.is_simulated_batch)
                self.assertEqual(X.shape, (bs, feat))
                self.assertEqual(y.shape, (bs,))
                seen_rows.append(X)
                seen_labels.append(y)

            X_all = torch.vstack(seen_rows)
            y_all = torch.hstack(seen_labels)

            # Expect to have seen N unique indices with no duplicates in one epoch
            self.assertEqual(len(torch.unique(y_all)), N)
            self.assertTrue(torch.equal(torch.sort(y_all).values, torch.arange(N)))

            # After consuming an epoch, the loader should reset (epoch increment)
            # NOTE: This assertion will FAIL if the internal step/epoch counters don’t advance.
            self.assertGreaterEqual(dl.static_epoch, 1, msg="static_epoch did not advance after one full pass")

    def test_OrchestratorDataset_mixed_static_and_sim(self):
        """
        With both static tensors and sim args provided, the loader should produce
        a mix of static and simulated batches following _method_selection_idxs
        derived from percent_static.
        """
        torch.manual_seed(RANDOM_SEED)

        N = 100
        bs = 10
        feat = 4

        data = torch.randn(N, feat)
        labels = torch.arange(N, dtype=torch.long)

        percent_static = 60.0  # expect exactly 60 static batches out of 100 steps

        # Minimal stand-ins for sim args
        dummy_sim_cfg = object()
        dummy_sim_data = object()

        def fake_run_simulation(_cfg, _data):
            return torch.zeros(bs, feat), torch.full((bs,), -1, dtype=torch.long)

        args = OrchestratorDatasetArgs(
            batch_size=bs,
            percent_static=percent_static,
            sim_config_args=dummy_sim_cfg,
            sim_data_args=dummy_sim_data,
            static_data=data,
            static_labels=labels,
        )

        with patch("cover_class.dataloader.run_simulation", side_effect=fake_run_simulation):
            dl = OrchestratorDataset(args, shuffle=True)

            static_count = 0
            sim_count = 0

            # Consume 100 steps to align with the size of _method_selection_idxs
            for _ in range(100):
                X, y = next(dl)
                if dl.is_simulated_batch:
                    sim_count += 1
                    # Simulated sentinel values
                    self.assertTrue(torch.all(X == 0))
                    self.assertTrue(torch.all(y == -1))
                else:
                    static_count += 1
                    # Static batch shape checks
                    self.assertEqual(X.shape, (bs, feat))
                    self.assertEqual(y.shape, (bs,))

            self.assertEqual(static_count, int(percent_static))
            self.assertEqual(sim_count, 100 - int(percent_static))
