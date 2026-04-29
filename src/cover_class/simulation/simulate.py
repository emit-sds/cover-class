from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from torch import (
    BoolTensor,  # boolean
    CharTensor,  # int8
    FloatTensor, # float32
    ShortTensor, # int16
    IntTensor,   # int32
    LongTensor,  # int64
    Tensor,
)
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch import device as Device

from cover_class.simulation.args import SimulationArgs, DataArgs, ForceFracRange


# The masked out alpha value will be 2^-126, the IEEE 754 smallest positive float32 value
ALPHA_MASKOUT_VALUE = torch.tensor(2 ** -126, dtype=torch.float32)
NULL_CLASS_VALUE = -1

def apply_glint_offset(spectra: FloatTensor, labels: Tensor, glint_constant_range: Tuple[Optional[float], Optional[float]], water_classes: List[int]) -> FloatTensor:
    lo, hi = glint_constant_range
    if lo is None or hi is None or (len(water_classes) == 0) or (lo == hi == 0.0): 
        return spectra
    assert lo <= hi, f"glint_scalar_range lower bound ({lo}) must be <= upper bound ({hi})"

    m = labels[..., None].eq(torch.as_tensor(water_classes, device=labels.device, dtype=labels.dtype)).any(-1)
    if not m.any(): 
        return spectra
    
    s = lo + (hi - lo) * torch.rand(spectra.size(0), device=spectra.device, dtype=spectra.dtype)
    spectra[m] += s[m].unsqueeze(-1)
    return spectra

def augment_magnitude(spectra: FloatTensor, labels: Tensor, class_scalar_ranges: List[Tuple[Optional[float], Optional[float]]], magnitude_max: Optional[float]) -> FloatTensor:
    for i, magnitude_range in enumerate(class_scalar_ranges):
        if magnitude_range is None or len(magnitude_range) == 0:
            continue

        lo, hi = magnitude_range
        if lo == hi == 1.0 or lo is None or hi is None:
            continue
        lo, hi = (hi, lo) if hi < lo else (lo, hi)

        m = labels[..., None].eq(i).any(-1)
        if not m.any():
            continue

        num = int(m.sum())
        s = lo + (hi - lo) * torch.rand(num, device=spectra.device, dtype=spectra.dtype)

        if magnitude_max is not None:
            spec_max = spectra[m].amax(dim=1)
            s = torch.where(spec_max > 0, torch.minimum(s, magnitude_max / spec_max), s)

        spectra[m] = spectra[m] * s.unsqueeze(-1)
    return spectra

def reduce_between(mask: BoolTensor, cumsum_n_components: LongTensor) -> ShortTensor:
    mask_pad = F.pad(mask.to(dtype=torch.int64, device=mask.device).cumsum_(dim=1), (1, 0), value=0) # pad left side with 0
    s = (
        mask_pad.gather(1, cumsum_n_components[:, 1:]) - 
        mask_pad.gather(1, cumsum_n_components[:, :-1])
    )
    return s.to(dtype=torch.int16, device=mask.device) # type: ignore


def run_simulation(
        sim_args:  SimulationArgs, 
        data_args: DataArgs, 
        device:    Device = Device("cpu"),
        force_class: Optional[int] = None,
        force_frac_range: Optional[ForceFracRange] = None,
    
    ) -> Tuple[FloatTensor, LongTensor, Optional[FloatTensor]]:

    sim_args.to(device)
    data_args.to(device)

    real_spectra = data_args.real_spectra.clone()
    real_labels  = data_args.real_labels
    real_spectra = apply_glint_offset(real_spectra, real_labels, sim_args.glint_constant_range, sim_args.water_classes) # type: ignore
    real_spectra = augment_magnitude(real_spectra, real_labels, sim_args.class_scalar_ranges, sim_args.magnitude_max)

    with torch.no_grad():
        classes, cumsum_n_components = _0_init_simulation_state(sim_args, device, force_class)

        simulation_space_size = (sim_args.n_iters, int(cumsum_n_components.max().item()))
        alpha = _1_generate_alpha(
            simulation_space_size,
            cumsum_n_components,
            sim_args.alpha,
            sim_args.alpha_uniform_low, 
            sim_args.alpha_uniform_high
        )
        
        # dirich_fractions shape: (n_iters X max(cumsum_n_components))
        # mask shape: (n_iters X max(cumsum_n_components))
        dirich_fractions, mask = _2_generate_dirichlet_distribution(alpha, sim_args.min_frac)
        del alpha

        filtered_n_components_per_class: ShortTensor
        if force_frac_range is not None:
            lo, hi = force_frac_range.low, force_frac_range.high
            if hi < lo:
                lo, hi = hi, lo

            # Number of components per selected class in each row.
            filtered_n_components_per_class = (cumsum_n_components[:, 1:] - cumsum_n_components[:, :-1]).to(dtype=torch.int16, device=device) #type: ignore

            # Sum the Dirichlet fractions by class for each row, then look at the target class.
            fracs_by_class = get_fractions_by_class(filtered_n_components_per_class, classes, dirich_fractions)

            keep_rows = ((fracs_by_class[:, force_class] >= lo) & (fracs_by_class[:, force_class] <= hi))

            classes = classes[keep_rows] #type: ignore
            cumsum_n_components = cumsum_n_components[keep_rows] #type: ignore
            dirich_fractions = dirich_fractions[keep_rows] #type: ignore
            filtered_n_components_per_class = filtered_n_components_per_class[keep_rows] #type: ignore
        else:
            dirich_fractions = _3_remove_small_fractions(dirich_fractions, mask)
            filtered_n_components_per_class = reduce_between(mask, cumsum_n_components)

            # if there are classes that're no longer present after the filtering, replace them with `NULL_CLASS_VALUE`
            classes.masked_fill_(~filtered_n_components_per_class.to(dtype=torch.bool, device=device), NULL_CLASS_VALUE)
        del mask

        selected_idxs, spectra_mask = _4_stratified_split(
            filtered_n_components_per_class,
            classes,
            data_args.real_labels,
            sim_args.n_classes,
            max([max(i) for i in sim_args.n_components]),
            simulation_space_size[1],
        )

        resulting_real_spectra = _5_make_sim_spectra(
            real_spectra,
            selected_idxs, 
            spectra_mask, 
            dirich_fractions
        )
        del selected_idxs, spectra_mask

        resulting_real_spectra += _6_add_noise(
            resulting_real_spectra, #type: ignore
            sim_args.noise_covariance,
            len(classes),
            real_spectra.shape[1], 
            sim_args.noise_scalar if sim_args.noise_scalar is not None else 1.,
            sim_args.white_noise,
            sim_args.noise_brightness_scaling,
            device
        )

        if sim_args.return_fractions:
            fracs_by_class = get_fractions_by_class(
                filtered_n_components_per_class,
                classes,
                dirich_fractions,
            )
            return resulting_real_spectra, classes.long(), fracs_by_class # type: ignore[return-value]
        return resulting_real_spectra, classes.long(), None # type: ignore[return-value]


def _0_init_simulation_state(
        sim_args:    SimulationArgs,
        device:      Device,
        force_class: Optional[int] = None,

    ) -> Tuple[
        CharTensor, # classes
        LongTensor  # cumsum_n_components
    ]:
    
    selections = torch.multinomial(
        sim_args.sim_mixture_probs, 
        num_samples=sim_args.n_iters, 
        replacement=True
    )
    classes = sim_args.sim_mixture_probs_matrix[selections]
    # Convert multi-hot rows like [0,1,0,1,1] -> class-id rows like [1,3,4,-1]
    max_active = int(classes.bool().sum(dim=1).max().item())

    class_ids = torch.arange(classes.size(1), device=classes.device).expand_as(classes)
    classes = torch.where(classes.bool(), class_ids, classes.size(1))
    classes = classes.sort(dim=1).values[:, :max_active]
    classes = classes.masked_fill(classes == sim_args.n_classes, NULL_CLASS_VALUE)

    size = classes.shape
    
    classes_cpu = classes.to(dtype=torch.int64).cpu().numpy()
    n_components_numpy: npt.NDArray[np.int32] = np.zeros(size, dtype=np.int32)
    for c in np.unique(classes_cpu):
        if c == NULL_CLASS_VALUE:
            continue
        mask = (classes_cpu == c)
        vals = np.asarray(sim_args.n_components[c], dtype=np.int32)
        n_components_numpy[mask] = np.random.choice(vals, mask.sum(), replace=True)
    n_components = torch.from_numpy(n_components_numpy).to(dtype=torch.int64, device=device)

    if force_class is not None:
        # Add in the force class to every row
        missing = ~(classes == force_class).any(dim=1)
        if missing.any():
            cols = torch.randint(low=0, high=size[1], size=(int(missing.sum().item()),), device=device)
            rows = missing.nonzero(as_tuple=False).squeeze(1)
            classes[rows, cols] = force_class
        
        # And force it to have 1 component (only draw 1 real sample)
        n_components[(classes == force_class)] = 1

    cumsum_n_components = F.pad(n_components.cumsum_(dim=1), (1, 0), value=0)  # pad left size with zeros

    return classes, cumsum_n_components # type: ignore[return-value]


def _1_generate_alpha(
        size:                Tuple[int, int],
        cumsum_n_components: LongTensor,
        a:                   Optional[float],
        a_uniform_low:       float,
        a_uniform_high:      float,
        
    ) -> FloatTensor:
    device = cumsum_n_components.device
    
    if a is not None:
        alpha = torch.full(size, a, dtype=torch.float32, device=device)
    else:
        alpha = (a_uniform_low + ((a_uniform_high - a_uniform_low) * torch.rand(size, dtype=torch.float32, device=device)))

    # Mask out all alpha values to 2^-126 if their index is greater than the number of components in their simulation
    # `cumsum_n_components` is substracted by 1 since `cumsum_n_components` values are essentially 1-indexed
    alpha_mask = torch.arange(alpha.shape[1], device=device).unsqueeze(0) > (cumsum_n_components[:, -1].view(-1, 1) - 1) 
    alpha.masked_fill_(alpha_mask, ALPHA_MASKOUT_VALUE)
    return alpha # type: ignore[return-value]


def _2_generate_dirichlet_distribution(
        alpha:    FloatTensor,
        min_frac: float,
        
    ) -> Tuple[
        FloatTensor, # dirich_fractions
        BoolTensor   # mask
    ]:

    dirich_fractions = Dirichlet(alpha).sample().to(alpha.device) # is already a float32
    mask = dirich_fractions >= min_frac
    return dirich_fractions, mask # type: ignore[return-value]


def _3_remove_small_fractions(
        dirich_fractions:    FloatTensor, 
        mask:                BoolTensor,
        
    ) -> FloatTensor:

    # all operations are inplace
    r = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    mask[r, dirich_fractions.argmax(dim=1)[r]] = True # guarantees at least 1 class per simulation

    dirich_fractions.masked_fill_(~mask, 0)
    dirich_fractions.div_(dirich_fractions.sum(dim=1)[:, None])

    # now we need to align the matrix again
    return dirich_fractions.gather(1, (dirich_fractions != 0).int().argsort(descending=True)) # type: ignore[return-value]


def _4_stratified_split(
        filtered_n_components_per_class: ShortTensor, 
        classes: CharTensor,
        labels: Tensor,
        n_classes: int,
        n_components_max: int,
        dirichlet_2nd_dim_shape: int,

    ) -> Tuple[
        IntTensor,   # selected_idxs
        BoolTensor   # spectra_mask
    ]:
        device = filtered_n_components_per_class.device

        ### 1. Make a (N_classes, N_classes_per_sim, N_labels) matrix that's a boolean mask for if each data label is a specific class
        unique_classes = torch.arange(n_classes, dtype=torch.int8, device=device)
        is_class       = (unique_classes[:, None] == labels[None, :]).to(device)
        assert (is_class.sum(dim=1) >= n_components_max).all(), f"All classes must have more than {n_components_max} labels"

        # (N_classes, N_labels) -> (N_iters, N_classes_per_sim, N_labels)
        # Note: So if the `NULL_CLASS_VALUE` is -1, then the mask will be `True` for where `labels` is the highest class value.
        #   This gets mitigated, though, downstream by the `filtered_n_components_per_class` filtering `selection_mask` to be 0 for those `NULL_CLASS_VALUE` classes
        is_class = is_class[classes.to(dtype=torch.long, device=device)]
        del unique_classes

        ### 2. Randomly sample the indices for each class, for each time they'll be in a simulated set
        selected_idxs = (torch.
            # (N_iters, N_classes_per_sim, n_components_max) selected indices
            rand(is_class.shape, device=device).
            masked_fill_(~is_class, 0).
            topk(n_components_max, dim=2).
            indices
        )
        del is_class

        # Only get the number of sample indices needed we're going to use for the simulation
        selection_mask = torch.arange(n_components_max, device=device)[None,None,:] < filtered_n_components_per_class[..., None]
        selected_idxs = (selected_idxs.
            # (N_iters, N_classes_per_sim X n_components_max) selected indices
            masked_fill_(~selection_mask, -1).
            reshape(selected_idxs.size(0), -1)
        )

        ### 3. Reshape the `selected_idxs` to the sameshape as `dirich_fractions` with all valid indices left-adjusted
        # (the issue is now we have a `selected_idxs` matrix of (N_iters, N_classes_per_sim, n_components_max) and we need (N_iters, max_cumsum_components))
        indices = (torch.
            # (N_iters, max_cumsum_components) - same shape as `dirich_fractions`
            where(
                selection_mask.reshape(selected_idxs.size(0), -1),
                torch.arange(selected_idxs.size(1), device=device).expand_as(selected_idxs), # (arange N_iters times)
                selected_idxs.size(1)
            ).
            topk(dirichlet_2nd_dim_shape, largest=False).
            indices
        )
        
        selected_idxs = (selected_idxs.gather(1, indices).to(dtype=torch.int32, device=device)) # (N_iters, max_cumsum_components) - properly shaped selected indices
        del indices, selection_mask

        
        spectra_mask = selected_idxs == -1 # mask out unwanted ids
        selected_idxs[spectra_mask] = 0
        
        return selected_idxs, spectra_mask # type: ignore[return-value]


def _5_make_sim_spectra(
        spectra:          FloatTensor,
        selected_idxs:    IntTensor, 
        spectra_mask:     BoolTensor,
        dirich_fractions: FloatTensor
        
    ) -> Tensor:

    resulting_real_spectra = ( # (N_iters, spectra_dim)
        # get the spectra and 0 out the unused spectra
        # (N_iters, max_cumsum_components, spectra_dim)
        spectra.
        index_select(0, selected_idxs.flatten()).
        reshape(*selected_idxs.shape, -1).
        masked_fill_(spectra_mask[..., None], 0).

        # now multiply by `dirich_fractions`
        # (N_iters, spectra_dim)
        mul_(dirich_fractions.unsqueeze(-1)).
        sum(dim=1)
    )
    return resulting_real_spectra


def _6_add_noise(
        spectra:           FloatTensor,
        sim_args_noise:    Optional[torch.FloatTensor],
        n_iters:           int,
        wavelength_dim:    int,
        noise_scalar:      float,
        white_noise_scale: float,
        noise_brightness_scaling: bool,
        device:            Device

    ) -> FloatTensor:
    
    if sim_args_noise is not None:
        if not (torch.linalg.eigvals(sim_args_noise).real>=0).all():
            sim_args_noise = make_positive_definite(sim_args_noise)
        means = torch.zeros(sim_args_noise.shape[0], dtype=torch.float32, device=device)
        noise = torch.distributions.MultivariateNormal(means, covariance_matrix=sim_args_noise * noise_scalar).sample((n_iters,))
        
        # add white noise
        white_noise = torch.normal(mean=means.expand(n_iters, -1), std=float(white_noise_scale))
        noise = noise + white_noise
        
    else:
        noise = torch.zeros(n_iters, wavelength_dim, dtype=torch.float32, device=device) * noise_scalar

    # Roughly scale noise by spectral mean
    if noise_brightness_scaling:
        spectral_mean = torch.mean(spectra, dim=1, keepdim=True)
        noise = noise * spectral_mean

    return noise.to(dtype=torch.float32)  # type: ignore[return-value]


def make_positive_definite(A: Tensor, min_eigen=1e-8) -> FloatTensor:
    assert len(A.shape)==2, "A needs to be a 2D matrix"
    A = (A + A.T) * 0.5 # makes real, symmetric
    eigenvals, eigenvecs = torch.linalg.eigh(A)
    eigenvals = torch.clamp(eigenvals, min=min_eigen)
    return (eigenvecs * eigenvals.unsqueeze(-2)) @ eigenvecs.T


def get_fractions_by_class(
        filtered_n_components_per_class: ShortTensor, 
        classes: CharTensor,
        dirich_fractions: FloatTensor,

    ) -> FloatTensor:
    # Returns a (n_iter, n_classes) matrix of the sum of dirichlet constributions per class
    
    # first get the dirichlet fractions by number of components
    n_iters, n_classes_per_sim = filtered_n_components_per_class.shape
    n_max_sim_comps = dirich_fractions.size(1)
    ends = filtered_n_components_per_class.cumsum(1) # (R, N)
    pos = torch.arange(n_max_sim_comps, device=dirich_fractions.device).unsqueeze(0).repeat(n_iters, 1)
    grp = torch.searchsorted(ends, pos, right=True)
    fmask = pos < ends[:, -1:]
    summed_fracs = dirich_fractions.new_zeros(n_iters, n_classes_per_sim)
    summed_fracs.scatter_add_(1, grp.clamp_max(n_classes_per_sim-1), dirich_fractions * fmask)

    # then assign each of those to a class (one-hot)
    valid = classes >= 0
    cls = classes.clamp_min(0)
    num_classes = int(cls[valid].max().item()) + 1 if valid.any() else 0
    result = summed_fracs.new_zeros(summed_fracs.size(0), num_classes)
    result.scatter_add_(1, cls.to(dtype=torch.int32), summed_fracs * valid)

    return result # type: ignore[return-value]