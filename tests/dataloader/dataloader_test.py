import unittest
import torch

from cover_class.dataloader import OrchestratorDataLoader, OrchestratorDataLoaderArgs # type: ignore[import]
from cover_class.simulation import SimulationArgs, DataArgs # type: ignore[import]

RANDOM_SEED = 42

class dataloaderTest(unittest.TestCase):

    def test_OrchestratorDataLoaderArgs(self):
        data = torch.randn(10, 5, dtype=torch.float32)
        labels = torch.randint(0, 3, (10,), dtype=torch.long)
        
        def make_odla(bsz, percent, s, d, sd, sl) -> OrchestratorDataLoaderArgs:
            return OrchestratorDataLoaderArgs(
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
    #     args = OrchestratorDataLoaderArgs(
    #         batch_size = 3,
    #         percent_static = 100.,
    #         static_data = static_data,
    #         static_labels = static_labels,
    #     )
    #     OrchestratorDataLoader()