import numpy as np
from scipy.interpolate import interp1d


# Function to test whether time samples of a time series are considered as missing or not:
# missing_fun_univ = np.isnan(s) # univariate
# missing_fun_multiv = lambda s: np.isnan(s).any(axis=1) # multivariate (the first dimension is time)
MISSING_FUN_DEFAULT = lambda s: np.isnan(s).any(axis=tuple(range(1, s.ndim))) # the first dimension is time


def interpolate_missing(x, missing_fun=MISSING_FUN_DEFAULT): # apply linear interpolation for missing value imputation
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    ind_missing = missing_fun(x)
    
    if x.ndim > 1: # multivariate
        return np.apply_along_axis(lambda o: interpolate_missing(o, missing_fun), 0, x)
    else: # univariate
        x_interp = x.copy()
        if np.isnan(x).all(): # all values missing - interp1d will not work
            # x_interp.fill(np.nan)
            pass
        elif sum(~np.isnan(x)) == 1: # only one non-missing value - interp1d will not work
            x_interp.fill(x[~np.isnan(x)][0]) # fill with the only non-missing value
        else:
            x_interp[np.where(ind_missing)[0]] = \
                interp1d(np.where(~ind_missing)[0], x[~ind_missing],
                        fill_value=(x[np.where(~ind_missing)[0][0]], x[np.where(~ind_missing)[0][-1]]), bounds_error=False # fill the missing parts at the beginning and at the end by the nearest non-missing value
                        )(np.where(ind_missing)[0])
        return x_interp