### static.yml
```
version: <version of the config [string]>
tags:    <various tags to descibe the config [list[string]]>
datasets:
  soil:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
  pv:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
  npv:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
  ash+char:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
  snow+ice:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
  water:
    location:    <paths to datasets [list[string]]>
    output-file: <output location [string]>
```

### dataloader.yml
```
version: <version of the config [string]>
tags:    <various tags to descibe the config [list[string]]>
drop-bands-wavelengths: <ranges of bands to drop [list[integer]]>
  - [400, 500]
  - [1000, 1200]
ood-test-set: <paths to OOD test HDF5 [string]>
datasets:
  soil:     <paths to datasets [list[string]]>
  pv:       <paths to datasets [list[string]]>
  npv:      <paths to datasets [list[string]]>
  ash+char: <paths to datasets [list[string]]>
  snow+ice: <paths to datasets [list[string]]>
  water:    <paths to datasets [list[string]]>
simulation:
  n_components:
    soil: [1,2,3]     <random choice for number of components per sample for class [list[integer]]>
    pv:   [1,2,3]     <random choice for number of components per sample for class [list[integer]]>
    npv:  [1,2,3,4,5] <random choice for number of components per sample for class [list[integer]]>
  n_classes_in_subsets: <number of classes in the simulated spectra [integer]>
  min_frac:             <minimum fraction of class presence to be included in simulation [float]>
  alpha_uniform_low:    <Dirichlet distribution alpha low value [float]>
  alpha_uniform_high:   <Dirichlet distribution alpha high value [float]>
  white_noise:          <white noise scalar [float]>
  noise_covariance_csv: <path to the noise covariance csv file [string]>
  return_fractions:     <return the dirichlet fractions from the simulation [boolean]>
  glint_upper_scalar:   <upper bound for water class glint scalar [float]>
  glint_lower_scalar:   <lower bound for water class glint scalar [float]>
subsample:
  test-fraction:   <test data split fraction [float]>
  selected-method: <subsampling method to use [string]>
  file-specific:
    /some/path.hdf5:
        kmedoids:
  convex-hull:
    ...
  kmeans:
    ...
  kmedoids:
    ...
  lhs:
    ...
dataloader:
  percent-static-data: 0.5
test-scene-urls:
    - https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/some-fid
```

All of the fields in each of the `subsample` methods in the `dataloader.yml` are the names of a parameter and the value of that parameter to be passed into the corresponding subsampling function