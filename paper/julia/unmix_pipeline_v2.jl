using CSV
using DataFrames
using Statistics
using Random
using HDF5
using JSON
using ProgressMeter
using SpectralUnmixing

const LABEL_RENAMES = Dict(
    "char-ash" => "ash-char",
    "rock" => "soil",
    "snow-ice" => "snow+ice",
)

normalize_label(x) = get(LABEL_RENAMES, String(x), String(x))
is_hdf5_file(path::AbstractString) = lowercase(splitext(path)[2]) in [".h5", ".hdf5"]
is_json_file(path::AbstractString) = lowercase(splitext(path)[2]) == ".json"

function read_h5_attrs(path::AbstractString)
    """Read root HDF5 attributes."""
    try
        return h5readattr(path, "/")
    catch err
        @warn "Could not read root HDF5 attributes" path exception=(err, catch_backtrace())
        return Dict{String,Any}()
    end
end

function first_attr(attrs, keys::Vector{String})
    """Return the first matching attribute value."""
    for key in keys
        if haskey(attrs, key)
            return attrs[key]
        end
    end
    return nothing
end

function vectorize_wavelengths(x)
    """Convert wavelength attributes to a Float64 vector."""
    x === nothing && return Float64[]
    return Float64.(vec(x))
end

function orient_spectra(spectra, wavelengths::Vector{Float64})
    """Orient spectra as rows and bands as columns."""
    X = Float64.(Array(spectra))
    ndims(X) == 1 && (X = reshape(X, 1, :))

    if !isempty(wavelengths)
        if size(X, 2) == length(wavelengths)
            return X
        elseif size(X, 1) == length(wavelengths)
            return permutedims(X)
        end
    end

    return X
end

function dataframe_from_spectra(X::AbstractMatrix, wavelengths::Vector{Float64})
    """Make a spectra DataFrame with wavelength column names."""
    if isempty(wavelengths)
        wavelengths = collect(1:size(X, 2)) .|> Float64
    end

    if size(X, 2) != length(wavelengths)
        error(
            "Spectra have $(size(X, 2)) bands but wavelength vector has " *
            "$(length(wavelengths)) entries"
        )
    end

    return DataFrame(X, string.(wavelengths))
end

function read_test_spectra(path::AbstractString, class_valid_keys::Vector{String})
    """Read test/OOD spectra from HDF5 or CSV."""
    if is_hdf5_file(path)
        attrs = read_h5_attrs(path)
        wavelengths = vectorize_wavelengths(
            first_attr(attrs, ["good_wavelengths", "wavelengths", "wl", "wave"])
        )

        X = orient_spectra(h5read(path, "spectra"), wavelengths)

        if isempty(wavelengths)
            wavelengths = collect(1:size(X, 2)) .|> Float64
        elseif size(X, 2) != length(wavelengths)
            good = vectorize_wavelengths(first_attr(attrs, ["good_wavelengths"]))
            if !isempty(good) && size(X, 2) == length(good)
                wavelengths = good
            else
                error("Test spectra bands do not match HDF5 wavelengths in $path")
            end
        end

        spectra_df = dataframe_from_spectra(X, wavelengths)
        fractions_df = DataFrame()

        has_labels = h5open(path, "r") do f
            return haskey(f, "labels")
        end

        if has_labels
            y = Array(h5read(path, "labels"))
            ndims(y) == 1 && (y = reshape(y, :, 1))

            if size(y, 2) != length(class_valid_keys) &&
               size(y, 1) == length(class_valid_keys)
                y = permutedims(y)
            end

            if size(y, 2) == length(class_valid_keys)
                fractions_df = DataFrame(Float64.(y), class_valid_keys)
            else
                @warn(
                    "Ignoring labels dataset because shape does not match",
                    size=size(y),
                    nclasses=length(class_valid_keys),
                )
            end
        end

        return spectra_df, fractions_df, wavelengths
    end

  df = DataFrame(CSV.File(path))
  rename!(df, Dict(c => normalize_label(c) for c in names(df)))
  
  sim_wl = [
      parse(Float64, s)
      for s in names(df)
      if !isnothing(tryparse(Float64, s))
  ]
  
  spectra_df = df[:, string.(sim_wl)]
  fraction_cols = [c for c in class_valid_keys if c in names(df)]
  fractions_df = isempty(fraction_cols) ? DataFrame() : df[:, fraction_cols]
  
  return spectra_df, fractions_df, sim_wl
end

function materialize_library_file(library_file::AbstractString, out_dir::AbstractString)
    """Validate that Julia receives a CSV unmixing library."""
    if lowercase(splitext(library_file)[2]) != ".csv"
        error(
            "library_file must be a CSV. Convert reduced-library HDF5 files " *
            "in Python before calling Julia: $library_file"
        )
    end

    isfile(library_file) || error("Library CSV not found: $library_file")
    return library_file
end

function zero_fractions(k::Int, class_valid_keys::Vector{String})
    """Create empty truth fractions when labels are unavailable."""
    return DataFrame(zeros(k, length(class_valid_keys)), class_valid_keys)
end

function load_class_subset(
    path::Union{Nothing,String},
    k::Int,
    class_valid_keys::Vector{String},
    force_full_classes::Bool,
)
    """Load per-spectrum class subsets or use all classes."""
    if force_full_classes || path === nothing || isempty(path) || !isfile(path)
        return [copy(class_valid_keys) for _ in 1:k]
    end

    parsed = JSON.parsefile(path)
    subsets = [String.(x) for x in parsed]

    if length(subsets) != k
        @warn(
            "class_subset length does not match spectra count; using all classes",
            n_subset=length(subsets),
            n_spectra=k,
        )
        return [copy(class_valid_keys) for _ in 1:k]
    end

    return [isempty(x) ? copy(class_valid_keys) : normalize_label.(x) for x in subsets]
end

function valid_subset(subset, class_valid_keys::Vector{String})
    """Keep only known classes and fall back to all if empty."""
    subset_str = [c for c in String.(subset) if c in class_valid_keys]
    isempty(subset_str) && return copy(class_valid_keys)
    return subset_str
end

function copy_truth_fractions!(
    simulated_mixtures::Matrix{Float64},
    simulated_fractions::DataFrame,
    i::Int,
    class_valid_keys::Vector{String},
)
    """Copy truth fractions by class name into full class order."""
    for (idx, c) in enumerate(class_valid_keys)
        if c in names(simulated_fractions)
            simulated_mixtures[i, idx] = Float64(simulated_fractions[i, c])
        end
    end
end

function fill_mixture_result!(
    mixture_results::Matrix{Float64},
    i::Int,
    mr,
    subset_str::Vector{String},
    class_valid_keys::Vector{String},
)
    """Map subset-order fractions into full class-order output."""
    mr = collect(mr)

    for (j, c) in enumerate(subset_str)
        idx = findfirst(==(c), class_valid_keys)
        if idx !== nothing && j <= length(mr) - 1
            mixture_results[i, idx] = Float64(mr[j])
        end
    end

    mixture_results[i, end] = Float64(mr[end])
end

function main(;
    n_mc,
    sim_file,
    out_dir,
    library_file,
    class_valid_keys,
    class_subset_file=nothing,
    model,
    num_endmembers,
    normalization,
    optimization,
    subsampling,
    max_combinations,
    combination_type,
    refl_scale=0.3,
    force_full_classes=false,
)
    mkpath(out_dir)

    if length(class_valid_keys) == 1 && isa(class_valid_keys[1], String)
        class_valid_keys = split(class_valid_keys[1], ",")
    end

    class_valid_keys = normalize_label.(String.(class_valid_keys))
    @info "Class keys" class_valid_keys

    simulated_library, simulated_fractions, sim_wl = read_test_spectra(
        sim_file,
        class_valid_keys,
    )

    k = size(simulated_library, 1)

    if isempty(names(simulated_fractions))
        simulated_fractions = zero_fractions(k, class_valid_keys)
    end

    @info "Loaded spectra" sim_file size=size(simulated_library)

    library_csv = materialize_library_file(library_file, out_dir)
    class_subset = load_class_subset(
        class_subset_file,
        k,
        class_valid_keys,
        force_full_classes,
    )

    mixture_results = zeros(k, length(class_valid_keys) + 1)
    simulated_mixtures = zeros(k, length(class_valid_keys))
    predicted_spectra = zeros(k, size(simulated_library, 2))
    comp_frac_results = Vector{Vector{Float64}}(undef, k)

    @showprogress for i in 1:k
        subset_str = valid_subset(class_subset[i], class_valid_keys)
        row = collect(simulated_library[i, :])

        copy_truth_fractions!(
            simulated_mixtures,
            simulated_fractions,
            i,
            class_valid_keys,
        )

        if length(subset_str) == 1
            class_idx_full = findfirst(==(subset_str[1]), class_valid_keys)

            if class_idx_full !== nothing
                mixture_results[i, class_idx_full] = 1.0
            end

            mixture_results[i, end] = 1.0
            comp_frac_results[i] = [1.0]
            predicted_spectra[i, :] = row
            continue
        end

        # Spectra and library are already on the 207 good-band grid, so no
        # regions are ignored. (With the default ignore list, the public
        # fork's get_good_bands_mask flags *every* band when no band lies
        # above 2440 nm, which NaNs the whole library.)
        unmixing_library = SpectralLibrary(library_csv, "label", 1, 1, subset_str, 1.0, Float64[])
        load_data!(unmixing_library)
        filter_by_class!(unmixing_library)
        interpolate_library_to_new_wavelengths!(unmixing_library, sim_wl)
        remove_wavelength_region_inplace!(unmixing_library, true)

        n_wl = length(unmixing_library.wavelengths)
        n_endmembers_i = min(
            num_endmembers,
            size(unmixing_library.spectra, 1),
            n_wl - 1,
        )

        if n_endmembers_i < 1
            error("No usable endmembers for pixel $i with classes $subset_str")
        end

        @assert size(unmixing_library.spectra, 2) == size(simulated_library, 2)

        class_idx = prepare_combinations(unmixing_library, combination_type)
        options = prepare_options(
            unmixing_library,
            combination_type,
            [n_endmembers_i],
            class_idx,
            subsample=true,
            max_combinations=max_combinations,
        )

        simulated_rfl = row ./ refl_scale
        simulated_rfl = reshape(simulated_rfl, (1, length(simulated_rfl)))

        mr, mv, cfr, cfv, cfru = unmix_pixel(
            unmixing_library,
            simulated_rfl,
            nothing,
            class_idx,
            options,
            model,
            n_mc,
            [n_endmembers_i],
            normalization,
            optimization,
            max_combinations,
            combination_type,
        )

        if any(isnan.(mr)) || any(isinf.(mr))
            @warn(
                "Invalid unmixing result; using equal selected-class fractions",
                pixel=i,
                classes=subset_str,
            )

            mr = zeros(length(subset_str) + 1)
            mr[1:length(subset_str)] .= 1.0 / length(subset_str)
            mr[end] = 1.0
        end

        fill_mixture_result!(
            mixture_results,
            i,
            mr,
            subset_str,
            class_valid_keys,
        )

        comp_frac_results[i] = cfr
        predicted_spectra[i, :] = unmixing_library.spectra' * cfru[1:end-1]
    end

    C = hcat(predicted_spectra, mixture_results[:, 1:end-1])
    col_names = vcat(string.(sim_wl), class_valid_keys)
    df = DataFrame(C, col_names)

    CSV.write(joinpath(out_dir, "predicted_fraction.csv"), df)
    return predicted_spectra
end

const CLI_TYPES = Dict(
    :n_mc => Int,
    :num_endmembers => Int,
    :max_combinations => Int,
    :refl_scale => Float64,
    :force_full_classes => Bool,
)

const CLI_DEFAULTS = Dict{Symbol,Any}(
    :class_subset_file => nothing,
    :combination_type => "class-even",
    :refl_scale => 0.3,
    :force_full_classes => false,
)

const CLI_REQUIRED = Set((
    :n_mc,
    :sim_file,
    :out_dir,
    :library_file,
    :class_valid_keys,
    :model,
    :num_endmembers,
    :normalization,
    :optimization,
    :subsampling,
    :max_combinations,
))

function parse_cli_value(k, v)
    """Parse one CLI value."""
    T = get(CLI_TYPES, k, String)

    if T === String
        return String(v)
    elseif T === Bool
        return lowercase(v) in ["true", "1", "yes", "y"]
    else
        return parse(T, v)
    end
end

function parse_args(argv=ARGS)
    """Parse simple --key value command-line arguments."""
    args = copy(CLI_DEFAULTS)
    i = 1

    while i <= length(argv)
        startswith(argv[i], "--") || error("Expected --option, got: $(argv[i])")
        key = Symbol(argv[i][3:end])
        i += 1

        if key == :class_valid_keys
            vals = String[]
            while i <= length(argv) && !startswith(argv[i], "--")
                append!(vals, split(argv[i], ","))
                i += 1
            end
            args[key] = filter(!isempty, vals)
        elseif key == :force_full_classes &&
               (i > length(argv) || startswith(argv[i], "--"))
            args[key] = true
        else
            i <= length(argv) || error("Missing value for --$key")
            args[key] = parse_cli_value(key, argv[i])
            i += 1
        end
    end

    missing = setdiff(CLI_REQUIRED, Set(keys(args)))
    if !isempty(missing)
        error(
            "Missing required CLI option(s): " *
            join("--" .* string.(collect(missing)), ", ")
        )
    end

    return args
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; parse_args()...)
end
