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
datasets:
  soil:     <paths to datasets [list[string]]>
  pv:       <paths to datasets [list[string]]>
  npv:      <paths to datasets [list[string]]>
  ash+char: <paths to datasets [list[string]]>
  snow+ice: <paths to datasets [list[string]]>
  water:    <paths to datasets [list[string]]>
simulation:
  n_components:         <possible numer of endmembers per class in the simulated spectra [list[integer]]>
  n_classes_in_subsets: <number of classes in the simulated spectra [integer]>
  alpha:                <alpha parameter for Dirichlet distribution [float]>
  min-frac:             <minimum fraction of class presence to be included in simulation [float]>
  noise-file:           <path to noise CSV file of covariances [string]>
subsample:
  test-fraction:   <test data split fraction [float]>
  selected-method: <subsampling method to use [string]>
  convex-hull:
    ...
  kmeans:
    ...
  kmedoids:
    ...
  lhs:
    ...
```

All of the fields in each of the `subsample` methods in the `dataloader.yml` are the names of a parameter and the value of that parameter to be passed into the corresponding subsampling function