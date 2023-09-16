<!---
[![DOI](<...>.svg)](https://<...>) 
-->


# computing **Dynamic Time Warping (DTW) distance** between time series with **missing data**

Developed by Aras Yurtman based on the [DTAIDistance library](https://github.com/wannesm/dtaidistance).

Corresponding paper:
> Aras Yurtman, Jonas Soenen, Wannes Meert, Hendrik Blockeel, "Estimating Dynamic Time Warping Distance Between Time Series with Missing Data," European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD) 2023. https://lirias.kuleuven.be/4091041?limo=0

## How to install

On this directory, run:

    python -m pip install .

## How to use:

### Proposed Method 1: <u>**DTW-AROW**</u> (DTW with Additional Restrictions on Warping)

A *local* method that only uses the two time series that are compared with each other.

It requires just the two time series (possibly with missing values) to compute the DTW distance between them.

The two time series that are compared can have different lengths and can be multivariate (with the same number of dimensions).

Example (see `notebooks/example_dtw_arow.ipynb`):

    import numpy as np
    import dtw_missing.dtw_missing as dtw_m
    import dtw_missing.dtw_missing_visualisation as dtw_m_vis

    x = np.array([0, 0, np.nan, np.nan, 1.7, 0.8, 0.4, 0.4, 1.1, 0, 0, 0.3, 0.3, 0, 0]) + 0.15
    y = np.array([0, 1, 1.5, 1.5, 1.3, 0, 0, np.nan, 1.1, 1.1, 0])

    # Calculate only the distance:
    d = dtw_m.warping_paths(y, x)[0]
    print(f'distance = {d}')

    # Calculate the distance and plot the warping:
    d, paths, best_path = dtw_m.warping_paths(y, x, return_optimal_warping_path=True)
    dtw_m_vis.plot_warpingpaths(y, x, paths, best_path, showlegend=True)
    dtw_m_vis.plot_warping(y, x, best_path, figsize=(4,3))


### Proposed Method 2: <u>**DTW-CAI**</u> (DTW with Clustering, Averaging, and Imputation)

A *global* method that uses not only the time series that are compared with each other, but also other time series in the dataset to leverage contextual information.

It requires a time series dataset to estimate the pairwise DTW distances between the instances.

The implementation requires a dataset that contains only univariate time series of the same length; however, it is straightforward to update it to handle multivariate time series of different lengths.

Example (see `notebooks/example_dtw_cai.ipynb`):

    import numpy as np
    import dtw_missing.experiments as exp

    # Create a simple dataset with missing values:
    dataset = np.array([[1, 1, np.nan, 5, 5, 2, 2], 
                        [1, 5, np.nan, np.nan, 5, 5, 2], 
                        [1.5, 5, 2, 2, 2, 2, np.nan], 
                        [np.nan, np.nan, np.nan, np.nan, 10, 10, 10]
                      ])
    
    # Both DTW-AROW and DTW-CAI parameters can be specified (because DTW-CAI uses DTW-AROW):
    dtw_params = {}
    dtw_cai_params = dict(no_clusters='elbow', no_clusters_range_for_elbow=(2,4)) 
        # ↑ limit the range because of the small dataset
    
    # Execute the DTW-CAI algorithm and get the pairwise distances:
    e_dtw_cai = exp.Experiment()
    e_dtw_cai.dataset = dataset
    e_dtw_cai.compute_pairwise_distances('dtw_cai', 
                                     missing_method_params=[dtw_params, dtw_cai_params],
                                     progress_bar=True, n_jobs=1)


### How to choose between DTW-AROW and DTW-CAI:

- DTW-AROW is much faster and does not require the full dataset.

- DTW-CAI performs more accurately in estimating DTW distances on the average for multiple datasets.

### Requirements
- dtaidistance >= 2.3.11
- numpy >= 1.20
- scipy >= 1.10
- joblib >= 1.1.0
- yellowbrick >= 1.5
- scikit-learn-extra >= 0.2.0

### License
Copyright 2021-2023 Aras Yurtman and Jonas Soenen, DTAI Research Group, KU Leuven.

Implementation of DTW-AROW (dtw_missing/dtw_missing.py) is based on dtaidistance: 
    https://github.com/wannesm/dtaidistance/blob/master/dtaidistance/dtw.py    

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.