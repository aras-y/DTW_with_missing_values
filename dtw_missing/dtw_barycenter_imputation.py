"""
missing value imputation using DBA (DTW Barycenter Averaging)
"""

import numpy as np
import copy
import warnings

from dtaidistance import dtw
import dtw_missing.dtw_missing as dtw_m

from joblib import Parallel, delayed
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# from dba_petitjean import DBA as dba_petitjean


def impute_with_dba_multiple(X, z, n_jobs=-1, progress_bar=False, **kwargs): # impute missing values in the array X of multiple time series using the barycentric average z, return imputed X
    X_imp = Parallel(n_jobs=n_jobs)(delayed(impute_with_dba)(x, z, **kwargs)
                                    for x in tqdm(X, desc='imputing', disable=not progress_bar)
                                   )
    
    # X_imp = []
    # for x in X:
    #     X_imp.append(impute_with_dba(x, z, **kwargs))
    
    return X_imp


def impute_with_dba(x, z, **kwargs): # impute missing values in x using the barycentric average z, return imputed x
    
    if "return_best_path" in kwargs:
        return_best_path = kwargs['return_best_path']
        del kwargs['return_best_path']
    
    d, paths, best_path = dtw_m.warping_paths(z, x, return_optimal_warping_path=True, **kwargs)
    
    z_warped = np.array(dtw.warp(z, x, path=best_path)[0])
    
    x_imp = copy.deepcopy(x)
    ind_nan = np.isnan(x)
    x_imp[ind_nan] = z_warped[ind_nan]
    
    return x_imp


# if __name__== "__main__": # test:
#     X = [np.array([0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0]) + 0.01,
#          np.array([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0]),
#          np.array([0, 0, 0, 0, 0.5, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0.5, 0.5, 0.5, 0.5]) + 0.1]
#
#     X_missing = copy.deepcopy(X)
#     X_missing[0][np.r_[4:6, 12:15, 21:23]] = np.nan
#     X_missing[1][np.r_[5:7, 12:17]] = np.nan
#
#     missing_method = 'restrict_diagonal_all'
#     window = None
#     psi = None
#     missing_normalization = 'avg_no_nonmissing_samples'
#
#     z = dba_petitjean.performDBA(X, n_iterations=10)
#     X_imp = impute_with_dba_multiple(X_missing, z, missing_method=missing_method, window=window, psi=psi, missing_normalization=missing_normalization)