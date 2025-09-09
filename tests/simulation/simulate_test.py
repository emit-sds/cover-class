import unittest
from unittest.mock import patch
import torch
from torch import Tensor

from cover_class.simulation import simulate # type: ignore[import]
from cover_class.simulation import SimulationArgs, DataArgs

RANDOM_SEED = 42

def new_SimulationArgs() -> SimulationArgs:
    return SimulationArgs(
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


class simulationTest(unittest.TestCase):
    def test_0_init_simulation_state(self):
        torch.manual_seed(RANDOM_SEED)
        sim_args = new_SimulationArgs()

        classes, cumsum_n_components = simulate._0_init_simulation_state(sim_args, None)

        self.assertEqual(classes.shape, (sim_args.n_iters, sim_args.n_classes_in_subsets))
        self.assertEqual(cumsum_n_components.shape, (sim_args.n_iters, sim_args.n_classes_in_subsets + 1))

        # 1. `classes` should be unique on a per-row basis
        # 2. All values in `classes` should be in range(n_classes)
        for r in classes:
            _, counts = torch.unique(r, return_counts=True)
            self.assertTrue((counts == 1).all().item())
        self.assertTrue((classes < sim_args.n_classes).all().item())
        self.assertTrue((0 <= classes).all())

        # 1. `cumsum_n_components` should be monotomically increasing on a per-row basis and should all be in sim_args.n_components
        differences = torch.diff(cumsum_n_components, axis=1)
        self.assertTrue((differences >= 0).all().item())
        self.assertTrue(torch.isin(torch.unique(differences), torch.tensor(sim_args.n_components)).all().item())
        # 2. `cumsum_n_components` have the first value in every row be zero
        self.assertTrue((cumsum_n_components[:, 0] == 0).all().item())


    def test_1_generate_alpha(self):
        torch.manual_seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            n_iters = 100
            cumsum_n_components: Tensor = torch.randint(1, 10, (n_iters, 10)) # type: ignore[annotation-unchecked]
            cumsum_n_components.cumsum_(dim=1)
            simulation_space_size = (n_iters, int(cumsum_n_components.max().item()))

            def check_alpha(alpha: Tensor, maskout_mask: Tensor):
                self.assertEqual(alpha.shape, simulation_space_size)
                bad = maskout_mask[:, :-1] & ~maskout_mask[:, 1:] # check for all simulate.ALPHA_MASKOUT_VALUE values being on the right side
                self.assertTrue(~bad.any(dim=1).any())

                for i in range(n_iters):
                    # assert the number of simulate.ALPHA_MASKOUT_VALUE values is equal to the (overall max cumsum value minus the last cumsum value in the row)
                    self.assertTrue(maskout_mask[i].sum() == (simulation_space_size[1] - cumsum_n_components[i][-1]))

            # condition 1: where no alpha is given - check the bounds
            alpha = simulate._1_generate_alpha(simulation_space_size, cumsum_n_components, None, 2, 10)
            self.assertTrue((alpha < 10).all())
            maskout_mask = torch.isclose(alpha, simulate.ALPHA_MASKOUT_VALUE)
            self.assertTrue((alpha[~maskout_mask] >= 2).all())
            check_alpha(alpha, maskout_mask)

            # condition 2: where alpha is given
            alpha = simulate._1_generate_alpha(simulation_space_size, cumsum_n_components, 0.5, 2, 10)
            maskout_mask = torch.isclose(alpha, simulate.ALPHA_MASKOUT_VALUE)
            self.assertTrue((alpha[~maskout_mask] == 0.5).all())
            check_alpha(alpha, maskout_mask)

        with self.subTest("test_function_receives_expected_input"):
            # the number of alphas needs to be the number of components to be simulated
            sim_args = new_SimulationArgs()
            sim_args.min_frac = float('inf')
            sim_args.alpha = 0.1
            data_args = DataArgs(torch.ones((120,10), dtype=torch.float32), torch.tensor(list(range(10))*12))

            received_alpha = None
            received_size = ()
            received_cumsum_tensor = torch.tensor([])
            classes, cumsum_n_components = simulate._0_init_simulation_state(sim_args, None)
            expected_size = (sim_args.n_iters, int(cumsum_n_components.max().item()))

            def mock_init_simulation_state(_: SimulationArgs, __):
                return classes, cumsum_n_components
            
            def mock_alpha(size, c, a, al, au):
                nonlocal received_size, received_cumsum_tensor, received_alpha
                received_size = size
                received_cumsum_tensor = c
                received_alpha = a
                return torch.full(size, 0.1, dtype=torch.float32)

            with (
                patch("cover_class.simulation.simulate._1_generate_alpha", side_effect=mock_alpha),
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_simulation_state)
            ):
                simulate.run_simulation(sim_args, data_args)
            self.assertEqual(received_size, expected_size)
            self.assertEqual(sim_args.alpha, received_alpha)
            torch.testing.assert_close(received_cumsum_tensor, cumsum_n_components)


    def test_2_generate_dirichlet_distribution(self):
        torch.manual_seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            # The dirichlet distribution needs to output values that sum up to 1 of the shape provided by alpha
            alpha = torch.rand((15, 5))
            dirichlet, mask = simulate._2_generate_dirichlet_distribution(alpha, 0)
            self.assertTrue(dirichlet.shape == alpha.shape)
            self.assertTrue((dirichlet.sum(dim=1) == 1).all())
            self.assertTrue(mask.all())

            a_mask = torch.where(torch.eye(alpha.size(1)) == 1)
            alpha[a_mask] = simulate.ALPHA_MASKOUT_VALUE
            dirichlet, mask = simulate._2_generate_dirichlet_distribution(alpha, 0)
            self.assertTrue(torch.isclose(dirichlet[a_mask], simulate.ALPHA_MASKOUT_VALUE, atol=1e-20, rtol=1e-20).all())

            _, mask = simulate._2_generate_dirichlet_distribution(alpha, 1)
            self.assertFalse(mask.any())

        with self.subTest("test_function_receives_expected_input"):
            # The dirichlet functions needs to be provided a number of alphas that correspond to the number of components to be simulated
            sim_args = new_SimulationArgs()
            data_args = DataArgs(torch.ones((120,10), dtype=torch.float32), torch.tensor(list(range(10))*12))

            received_alphas = torch.tensor([])
            received_min_frac = None
            classes, cumsum_n_components = simulate._0_init_simulation_state(sim_args, None)
            m = int(cumsum_n_components.max().item())

            def mock_init_simulation_state(_: SimulationArgs, __):
                return classes, cumsum_n_components

            def mock_dirichlet(alpha, min_frac):
                nonlocal received_alphas, received_min_frac, m, sim_args
                received_alphas = alpha
                received_min_frac = min_frac
                return torch.zeros((sim_args.n_iters, m)), torch.zeros((sim_args.n_iters,m), dtype=torch.bool)
            
            with (
                patch("cover_class.simulation.simulate._2_generate_dirichlet_distribution", side_effect=mock_dirichlet),
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_simulation_state)
            ):
                simulate.run_simulation(sim_args, data_args)
            self.assertEqual(received_alphas.shape, (sim_args.n_iters, m))
            self.assertEqual(received_min_frac, sim_args.min_frac)


    def test_3_remove_small_fractions(self):
        dirich_fractions = torch.ones((10, 10), dtype=torch.float32) * 0.1
        mask = torch.zeros((10, 10), dtype=torch.bool)
        mask[:, :2] = True
        simulate._3_remove_small_fractions(dirich_fractions, mask)
        self.assertTrue((dirich_fractions[:, :2] == 0.5).all().item())
        self.assertTrue((dirich_fractions[:, 2:] == 0).all().item())

        mask[:, :2] = False
        simulate._3_remove_small_fractions(dirich_fractions, mask)
        self.assertTrue(mask.sum(), mask.shape[0])
        self.assertTrue((dirich_fractions == 1).sum(), mask.shape[0])


    def test_4_stratified_split(self):
        torch.manual_seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            real_labels  = torch.tensor([0,0,0, 1,1,1,1, 2,2,2, 3,3]*4, dtype=torch.int8)
            classes      = torch.tensor([[0, 1], [2, 3]], dtype=torch.int8)
            n_components = torch.tensor([[2, 3], [4, 5]], dtype=torch.int16)
            m = int(n_components.max().item())
            d = int(n_components.sum(dim=1).max().item())

            selected_idxs, spectra_mask = simulate._4_stratified_split(
                n_components, classes, real_labels, len(torch.unique(real_labels)), m + 3, d
            )

            self.assertEqual(selected_idxs.shape, (n_components.shape[0], d))
            self.assertEqual(spectra_mask.shape, (n_components.shape[0], d))

            for i in range(len(n_components)):
                self.assertEqual((~spectra_mask[i]).sum(), n_components[i].sum())

                expected_classes = classes[i]
                expected_num_classes = n_components[i]
                these_selected_idxs = selected_idxs[i][~spectra_mask[i]]
                these_labels, counts = torch.unique(real_labels[these_selected_idxs], return_counts=True)

                self.assertListEqual(expected_classes.tolist(), these_labels.tolist())
                self.assertListEqual(these_labels.tolist(), expected_classes.tolist())
                self.assertListEqual(counts.tolist(), expected_num_classes.tolist())

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
                real_spectra=torch.tensor([
                        [10.0, 0.0, 0.0],
                        [11.0, 0.0, 0.0],
                        [12.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                        [21.0, 0.0, 0.0],
                        [22.0, 0.0, 0.0],
                    ],dtype=torch.float32,),
                real_labels=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int8),
            )

            classes = torch.tensor([[1, 2]], dtype=torch.int8)
            cumsum_n_components = torch.tensor([[1, 3]], dtype=torch.int32)

            def mock_init_sim_state(_: SimulationArgs, __):
                return classes, cumsum_n_components
            
            captured = {"filtered_n_components_per_class": None, "classes": None, "labels": None, "n_classes": None,
                        "n_components_max": None, "dirichlet_2nd_dim_shape": None}

            def mock_split(f, c, l, nc, ncm, d):
                captured["filtered_n_components_per_class"] = f
                captured["classes"] = c
                captured["labels"] = l
                captured["n_classes"] = nc
                captured["n_components_max"] = ncm
                captured["dirichlet_2nd_dim_shape"] = d
                return (
                    torch.zeros((sim_args.n_iters, cumsum_n_components.max().item()), dtype=torch.int32),
                    torch.ones((sim_args.n_iters, cumsum_n_components.max().item()), dtype=torch.bool),
                )

            with (
                patch("cover_class.simulation.simulate._0_init_simulation_state", side_effect=mock_init_sim_state),
                patch("cover_class.simulation.simulate._4_stratified_split", side_effect=mock_split),
            ):
                simulate.run_simulation(sim_args, data_args)

            for v in captured.values():
                self.assertIsNotNone(v)
            torch.testing.assert_close(captured["classes"], classes)
            torch.testing.assert_close(captured["labels"], data_args.real_labels)
            self.assertEqual(captured["n_classes"], sim_args.n_classes)
            self.assertEqual(captured["n_components_max"], max(sim_args.n_components))
            self.assertEqual(captured["dirichlet_2nd_dim_shape"], cumsum_n_components.max().item())

    def test_5_make_sim_spectra(self):
        torch.manual_seed(RANDOM_SEED)

        spectra = torch.rand((100, 20), dtype=torch.float32)
        n_iters, c = 10, 4
        sz = (n_iters, c)
        spectra_idxs = torch.randint(0, spectra.size(1), sz)
        spectra_mask = torch.randint(0, 2, sz, dtype=torch.bool)
        dirichlet = torch.rand(sz, dtype=torch.float32)

        resulting_spectra = simulate._5_make_sim_spectra(
            spectra,
            spectra_idxs,
            spectra_mask,
            dirichlet
        )

        self.assertEqual(resulting_spectra.shape, (n_iters, spectra.shape[1]))

        for i in range(n_iters):
            s = spectra[spectra_idxs[i]]
            d = dirichlet[i]
            ds = (s * d[:, None])[~spectra_mask[i]]
            if (~spectra_mask[i]).any():
                torch.testing.assert_close(ds.sum(dim=0), resulting_spectra[i])

    def test_6_add_noise(self):
        torch.manual_seed(RANDOM_SEED)

        with self.subTest("test_function_works_as_expected"):
            n_components = 10
            noise = simulate._6_add_noise(None, n_components, 0., None)
            torch.testing.assert_close(noise, torch.zeros(n_components))

            N = 10
            cov = torch.eye(N)
            noise = simulate._6_add_noise(cov, n_components, 0.4, None)
            self.assertEqual(noise.shape, (N,))

        with self.subTest("test_function_receives_expected_input"):
            N = 10
            cov = torch.eye(N)

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
            data_args = DataArgs(torch.ones((120,N), dtype=torch.float32), torch.tensor(list(range(10))*12))

            obtained_noise_cov, obtained_white_noise, obtained_wavelength_dim = None, None, None
                                    
            def mock_add_noise(sim_args_noise, wavelength_dim, white_noise_scale, _):
                nonlocal obtained_noise_cov, obtained_white_noise, obtained_wavelength_dim
                obtained_noise_cov = sim_args_noise
                obtained_white_noise = white_noise_scale
                obtained_wavelength_dim = wavelength_dim
                return torch.zeros(wavelength_dim)
            
            with patch("cover_class.simulation.simulate._6_add_noise", side_effect=mock_add_noise):
                simulate.run_simulation(sim_args, data_args)

            self.assertIsNotNone(obtained_noise_cov)
            self.assertIsNotNone(obtained_white_noise)
            self.assertIsNotNone(obtained_wavelength_dim)
            torch.testing.assert_close(obtained_noise_cov, cov)
            self.assertEqual(obtained_white_noise, sim_args.white_noise)
            self.assertEqual(obtained_wavelength_dim, data_args.real_spectra.shape[1])



if __name__ == "__main__":
    unittest.main()