# Disclaimer
`git lfs` is used to track these files. See [git-lfs](https://git-lfs.github.com/) before cloning or pulling.

# Original data
*__Edit : __ Data-sets were deleted due to their size.*
## Coordinates
Coordinates of cells can be found in `df_lonlat.csv`. The coordinates are the longitude and the latitude.

## Maxima of temperature
The temperature data can be found in `df_tas_cmip6_hadcrut_yearmax.csv`.

## Maxima of precipitation
The precipitation data can be found in `df_pr_cmip56_yearmax.csv`.

# Data-sets for learning
## Learning a distribution
The precipitation data used to learn can be found in `real_ds_dist_pluiv_glob_scaled.npy`. The non-scaled version can be found in `real_ds_dist_pluiv_glob.npy`.
The generated precipitation alike data can be found in `ds_dist_pluiv_glob.npy`.

## Learning a temporal tendency
The temperature data used to learn can be found in `real_ds_trend_temp_mod.npy`.
The generated temperature alike data can be found in `ds_trend_2_temp_mod.npy`.

## Learning a temporal and spatial structure
The temperature data used to learn can be found in `real_ds_cond_trend_temp_glob.npy`, with the associated labels in `real_lab_cond_trend_temp_glob.npy`.
The generated temperature alike data can be found in `ds_cond_trend_2_temp_glob.npy`, with the associated labels in `lab_cond_trend_2_temp_glob.npy`.
