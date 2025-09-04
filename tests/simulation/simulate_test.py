import unittest
from unittest.mock import patch
import numpy as np

from cover_class.simulation import simulate # type: ignore[import]
from cover_class.simulation import SimulationArgs, DataArgs

RANDOM_SEED = 42

class simulationTest(unittest.TestCase):
    def test_0_init_simulation_state(self):
        np.random.seed(RANDOM_SEED)
        sim_args = SimulationArgs(
            n_iters = 100,
            n_classes_in_subsets = 5,
            n_classes = 10,
            n_components = list(range(10)),
            min_frac = 0.,
            alpha = None,
            alpha_uniform_low = 0.,
            alpha_uniform_high = 0.,
            white_noise = 0.,
            noise_covariance = None
        )
        classes, total_n_components, total_n_comp_idxs = simulate._0_init_simulation_state(sim_args)

        self.assertEqual(classes.shape, (sim_args.n_iters, sim_args.n_classes_in_subsets))
        self.assertEqual(total_n_comp_idxs.shape, (sim_args.n_iters, sim_args.n_classes_in_subsets - 1))
        self.assertEqual(total_n_components.shape, (sim_args.n_iters,))

        # 1. `classes` should be unique on a per-row basis
        # 2. All values in `classes` should be in range(n_classes)
        for r in classes:
            _, counts = np.unique(r, return_counts=True)
            self.assertTrue((counts == 1).all())
        self.assertTrue((classes < sim_args.n_classes).all())
        self.assertTrue((0 <= classes).all())

        # 1. `total_n_comp_idxs` should be monotomically increasing on a per-row basis
        differences = np.diff(total_n_comp_idxs, axis=1)
        self.assertTrue((differences >= 0).all())

        # 1. `total_n_components` should be mostly be greater than the last value of the `total_n_comp_idxs` rows
        self.assertTrue((total_n_components > total_n_comp_idxs[:, -1]).any())


    def test_1_generate_alpha(self):
        np.random.seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            alpha = simulate._1_generate_alpha(None, 2, 10, 100)
            self.assertEqual(len(alpha), 100)
            self.assertTrue((alpha < 10).all())
            self.assertTrue((alpha >= 2).all())

            alpha = simulate._1_generate_alpha(0.1, 2, 10, 25)
            self.assertEqual(len(alpha), 25)
            self.assertTrue((alpha == 0.1).all())

        with self.subTest("test_function_receives_expected_input"):
            # the number of alphas needs to be the number of components to be simulated
            sim_args = SimulationArgs(
                n_iters = 100,
                n_classes_in_subsets = 5,
                n_classes = 10,
                n_components = list(range(10)),
                min_frac = float('inf'),
                alpha = None,
                alpha_uniform_low = 0.,
                alpha_uniform_high = 0.,
                white_noise = 0.,
                noise_covariance = None
            )
            data_args = DataArgs(np.ones((10,10)), np.ones((10,)))

            recieved_length = 0
            classes, total_n_components, total_n_comp_idxs = simulate._0_init_simulation_state(sim_args)

            def mock_init_simulation_state(sim_args: SimulationArgs):
                return classes, total_n_components, total_n_comp_idxs
            
            def mock_alpha(alpha, a, b, l):
                nonlocal recieved_length
                recieved_length = l
                return np.ones(l) * 0.1

            with (
                patch("cover_class.simulation.simulate._1_generate_alpha", side_effect=mock_alpha),
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_simulation_state)
            ):
                simulate.run_simulation(sim_args, data_args)
            self.assertEqual(recieved_length, total_n_components.sum())


    def test_2_generate_dirichlet_distribution(self):
        np.random.seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            # The dirichlet distribution needs to output values that sum up to 1 of the shape provided by alpha
            alpha = np.random.rand(15)
            dirichlet, mask = simulate._2_generate_dirichlet_distribution(alpha, 0)
            self.assertTrue(dirichlet.shape == alpha.shape)
            self.assertTrue(dirichlet.sum() == 1)
            self.assertTrue(mask.all())

            _, mask = simulate._2_generate_dirichlet_distribution(alpha, 1)
            self.assertFalse(mask.any())

        with self.subTest("test_function_receives_expected_input"):
            # The dirichlet functions needs to be provided a number of alphas that correspond to the number of components to be simulated
            sim_args = SimulationArgs(
                n_iters = 100,
                n_classes_in_subsets = 5,
                n_classes = 10,
                n_components = list(range(10)),
                min_frac = 0.,
                alpha = None,
                alpha_uniform_low = 0.,
                alpha_uniform_high = 0.,
                white_noise = 0.,
                noise_covariance = None
            )
            data_args = DataArgs(np.ones((10,10)), np.ones((10,)))

            num_alphas = 0
            classes, total_n_components, total_n_comp_idxs = simulate._0_init_simulation_state(sim_args)

            def mock_init_simulation_state(sim_args: SimulationArgs):
                return classes, total_n_components, total_n_comp_idxs

            def mock_dirichlet(alpha, min_frac):
                nonlocal num_alphas
                num_alphas += len(alpha)
                return np.zeros(1), np.zeros(1).astype(np.bool_)
            
            with (
                patch("cover_class.simulation.simulate._2_generate_dirichlet_distribution", side_effect=mock_dirichlet),
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_simulation_state)
            ):
                simulate.run_simulation(sim_args, data_args)
            self.assertEqual(num_alphas, total_n_components.sum())


    def test_3_remove_small_fractions(self):
        # The _3_remove_small_fractions should recieve a 1D array
        dirich_fractions = np.ones((10,)) * 0.1
        mask = np.zeros((10,)).astype(np.bool_)
        mask[:2] = True
        survivors = simulate._3_remove_small_fractions(dirich_fractions, mask)
        self.assertEqual(survivors.shape, (2,)) # 1 D
        self.assertTrue(survivors[0] == 0.5)
        self.assertTrue(survivors[1] == 0.5)

        with self.subTest("test_function_receives_expected_input"):
            sim_args = SimulationArgs(
                n_iters=1,
                n_classes_in_subsets=2,
                n_classes=3,
                n_components=[1, 2, 3],
                min_frac=0.0,
                alpha=0.3,
                alpha_uniform_low=0.0,
                alpha_uniform_high=0.0,
                white_noise=0.0,
                noise_covariance=None,
            )
            data_args = DataArgs(np.ones((10, 5)), np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 3]))

            dirich_fractions_received = None
            def mock_remove_small_fractions(_dirich_fractions, _mask):
                nonlocal dirich_fractions_received
                dirich_fractions_received = _dirich_fractions
                return _dirich_fractions
            
            with patch("cover_class.simulation.simulate._3_remove_small_fractions", side_effect=mock_remove_small_fractions):
                simulate.run_simulation(sim_args, data_args)

            self.assertIsNotNone(dirich_fractions_received)
            self.assertTrue(len(dirich_fractions_received.shape), 1) # 1D


    def test_4_stratified_split(self):
        np.random.seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            N, D = 12, 4
            real_labels  = np.array([0,0,0, 1,1,1,1, 2,2,2, 3,3], dtype=np.uint8)
            real_spectra = np.random.randn(N, D)
            classes      = np.array([0, 1], dtype=np.uint8)
            n_components = np.array([2, 3], dtype=np.uint16)

            spectra_subset, label_subset = simulate._4_stratified_split(
                real_spectra, real_labels, classes, n_components
            )

            self.assertEqual(spectra_subset.shape[0], n_components.sum())
            self.assertEqual(label_subset.shape[0], n_components.sum())

            for c, n in zip(classes, n_components):
                self.assertEqual((label_subset == c).sum(), n)

        with self.subTest("test_function_receives_expected_input"):
            sim_args = SimulationArgs(
                n_iters=1,
                n_classes_in_subsets=2,
                n_classes=5,
                n_components=[1, 2, 3],
                min_frac=0.0,
                alpha=0.3,
                alpha_uniform_low=0.0,
                alpha_uniform_high=0.0,
                white_noise=0.0,
                noise_covariance=None,
            )

            data_args = DataArgs(
                real_spectra=np.array([
                        [10.0, 0.0, 0.0],
                        [11.0, 0.0, 0.0],
                        [12.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                        [21.0, 0.0, 0.0],
                        [22.0, 0.0, 0.0],
                    ],dtype=np.float32,),
                real_labels=np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8),
            )

            classes_init = np.array([[1, 2]], dtype=np.uint8)
            total_n_components = np.array([3], dtype=np.uint16)
            total_n_comp_idxs = np.array([[1]], dtype=np.uint16)

            def mock_init_sim_state(_sim_args):
                return classes_init, total_n_components, total_n_comp_idxs
            
            captured = {"real_spectra": None, "real_labels": None, "classes": None, "n_components": None}

            def mock_split(real_spectra, real_labels, classes_recv, n_components_recv):
                captured["real_spectra"] = real_spectra
                captured["real_labels"] = real_labels
                captured["classes"] = classes_recv
                captured["n_components"] = n_components_recv
                return (
                    np.zeros((n_components_recv.sum(), real_spectra.shape[1]), dtype=np.float32),
                    np.ones(n_components_recv.sum()),
                )

            with (
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_sim_state),
                patch("cover_class.simulation.simulate._4_stratified_split", side_effect=mock_split),
            ):
                simulate.run_simulation(sim_args, data_args)

            self.assertIsNotNone(captured["real_spectra"])
            self.assertIsNotNone(captured["real_labels"])
            self.assertIsNotNone(captured["classes"])
            self.assertIsNotNone(captured["n_components"])
            np.testing.assert_array_equal(captured["real_spectra"], data_args.real_spectra)
            np.testing.assert_array_equal(captured["real_labels"], data_args.real_labels)
            np.testing.assert_array_equal(captured["classes"], np.array([1, 2], dtype=np.uint16))
            np.testing.assert_array_equal(captured["n_components"], np.array([1, 2], dtype=np.uint16))


    def test_5_add_noise(self):
        np.random.seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            n_components = 10
            noise = simulate._5_add_noise(None, n_components, 0.)
            np.testing.assert_array_equal(noise, np.repeat(0, n_components))

            N,D = 10, 5
            cov = np.cov(np.random.randn(N, D))
            noise = simulate._5_add_noise(cov, n_components, 0.4)
            self.assertEqual(noise.shape, (N,))

            noise = simulate._5_add_noise(np.ones(N), n_components, 0.4)
            self.assertEqual(noise.shape, (N,))


        with self.subTest("test_function_receives_expected_input"):
            cov = np.cov(np.random.randn(N, D))

            sim_args = SimulationArgs(
                n_iters=1,
                n_classes_in_subsets=2,
                n_classes=3,
                n_components=[1, 2, 3],
                min_frac=0.0,
                alpha=0.3,
                alpha_uniform_low=0.0,
                alpha_uniform_high=0.0,
                white_noise=123.45,
                noise_covariance=cov,
            )
            data_args = DataArgs(np.ones((10, D)), np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))

            obtained_noise_cov, obtained_white_noise, obtained_wavelength_dim = None, None, None
                                    
            def mock_add_noise(sim_args_noise, wavelength_dim, white_noise_scale):
                nonlocal obtained_noise_cov, obtained_white_noise, obtained_wavelength_dim
                obtained_noise_cov = sim_args_noise
                obtained_white_noise = white_noise_scale
                obtained_wavelength_dim = wavelength_dim
                return np.repeat(0, wavelength_dim)
            
            with patch("cover_class.simulation.simulate._5_add_noise", side_effect=mock_add_noise):
                simulate.run_simulation(sim_args, data_args)

            self.assertIsNotNone(obtained_noise_cov)
            self.assertIsNotNone(obtained_white_noise)
            self.assertIsNotNone(obtained_wavelength_dim)
            np.testing.assert_array_equal(obtained_noise_cov, cov)
            self.assertEqual(obtained_white_noise, sim_args.white_noise)
            self.assertEqual(obtained_wavelength_dim, data_args.real_spectra.shape[1])




if __name__ == "__main__":
    unittest.main()