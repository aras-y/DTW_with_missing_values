# Test dtw_missing for univariate and multivariate time series

import logging
import pytest

import numpy as np

import dtw_missing.dtw_missing as dtw_m
import dtw_missing.dtw_missing_visualisation as dtw_m_vis

from dtaidistance import dtw

from scipy.spatial import distance as scpydst


logger = logging.getLogger("be.kuleuven.dtai.dtw_missing")


def get_dtw_params_default():
    # Get default DTW parameters (both for DTW from dtaidistance and DTW-AROW for dtw_missing):
    dtw_params_default = {
        'window' : None,
        'max_dist' : None,
        'use_pruning' : False,
        'max_step' : None,
        'max_length_diff' : None,
        'penalty' : None,
        'psi' : None, #(2, 3, 4, 5), # (begin series1, end series1, begin series2, end series2)
        'psi_neg' : True,
        'use_c' : False,
        'use_ndim' : False,
    }
    return dtw_params_default.copy()


def get_dtw_params_partialrestrictions():
    # Get parameters for DTW with partial restrictions (i.e., a relaxed version of DTW-AROW):
    dtw_params = get_dtw_params_default()
    dtw_params.update({'missing_value_restrictions': 'partial'})
    return dtw_params


def get_default_univariate_time_series_simple_with_d():
    # Get simple default univariate time series:
    x = np.array([1, 2, 10, 1, 1], dtype=np.float32)
    y = np.array([1, 1, 2, 2, 13, 13, 1, 1, 1], dtype=np.float32)
    d = np.sqrt(2 * (3 ** 2))
    return x, y, d


def get_default_univariate_time_series():
    # Get default univariate time series:
    
    # x = np.array([0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0]) + 0.01
    # x = np.array([0, 0.3, 0.7, 1.5, 1.3, 0.1, 0, 0, 0.9, 0.7, 0, 0]) + 0.01
    x = np.array([0, 0, 1, 1.5, 1.7, 0.8, 0.4, 0.4, 1.1, 0, 0, 0.3, 0.3, 0, 0]) + 0.15
    # x = np.array([0, 0, 1, 1.5, 1.7, 0.8, 0.4, 0.4, 1.1, 0, 0, 0.5, 0]) + 0.15
    # x = np.array([0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0]) + 0.01
    # x = np.array([3, 4]) + 10

    # y = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
    # y = np.array([0, 0.1, 0.7, 1.5, 1, 0, 0.8, 0.8, 0.8, 0.8, 0.5, 0])
    y = np.array([0, 1, 1.5, 1.5, 1.3, 0, 0, 1.1, 1.1, 1.1, 0])
    # y = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.7, 1.3, 1.5, 1.3, 0.7, 0.3, 0.1, 0])
    # y = np.array([0, 0]) + 10

    # x = np.array([1, 2, 3])
    # y = np.array([0, 0, 0])
    
    return x, y


def convert_univariate_into_multivariate(x, y, D=2):
    # Convert a pair of univariate time series into multivariate ones (with D dimensions) by preserving the DTW distance between them:
    x = np.tile(x/np.sqrt(D), (D,1)).T
    y = np.tile(y/np.sqrt(D), (D,1)).T
    
    a = 10*np.random.rand(D) - 5 # constants to add to every dimension
    x += a
    y += a
    
    return x, y


def test_consistency_of_dtw_and_dtw_missing():
    # Test consistency of DTW from dtaidistance and DTW from dtw_missing for time series without missing values: 
    def test_consistency(x, y):
        d1 = dtw.warping_paths(x, y)[0]
        d2 = dtw_m.warping_paths(x, y)[0]
        assert d1 == d2
    
    x, y, _ = get_default_univariate_time_series_simple_with_d()
    test_consistency(x, y)
    
    x, y = get_default_univariate_time_series()
    test_consistency(x, y)


def test_dtw():
    # Test DTW-AROW on time series for known distances: 
    x, y, d_exp = get_default_univariate_time_series_simple_with_d()
    d = dtw.warping_paths(x, y)[0]
    assert d == pytest.approx(d_exp)


def test_dtw_arow():
    # Test DTW-AROW on time series with missing data for known distances: 
    x = np.array([1, 1, 2, 2, 10, 1, 1], dtype=np.float32)
    y = np.array([1, 1, 1, 2, 2, 13, 13, 1, 1, 1], dtype=np.float32)
    y[2:4] = np.nan
    d_exp = np.sqrt(2 * (3 ** 2))
    
    d = dtw_m.warping_paths(x, y, missing_value_adjustment=None)[0]
    assert d == pytest.approx(d_exp)
    
    d = dtw_m.warping_paths(x, y)[0]
    d_exp = d_exp * np.sqrt((len(x) + len(y)) / (np.sum(~np.isnan(x)) + np.sum(~np.isnan(y))))
    assert d == pytest.approx(d_exp)


def test_dtw_with_partialrestrictions():
    # Test DTW with partial restrictions (i.e., a relaxed version of DTW-AROW) for known distances: 
    x = np.array([1, 1, 2, 2, 10, 1, 1], dtype=np.float32)
    y = np.array([1, 1, 1, 2, 2, 13, 13, 1, 1, 1], dtype=np.float32)
    y[2:4] = np.nan
    d_exp = np.sqrt(2 * (3 ** 2))
    
    d = dtw_m.warping_paths(x, y, missing_value_restrictions='partial', missing_value_adjustment=None)[0]
    assert d == pytest.approx(d_exp)
    
    d = dtw_m.warping_paths(x, y, missing_value_restrictions='partial')[0]
    d_exp = d_exp * np.sqrt((len(x) + len(y)) / (np.sum(~np.isnan(x)) + np.sum(~np.isnan(y))))
    assert d == pytest.approx(d_exp)


def test_impossible_warping_in_dtw_arow_and_dtw_with_partialrestrictions():
    # Test DTW-AROW in a case where there is no warping that satisfies the restrictions:
    x, y = get_default_univariate_time_series()
    xL = len(x)
    yL = len(y)
    x[np.r_[0:2, 8:xL]] = np.nan
    y[np.r_[0:3, 7:yL]] = np.nan
    
    d = dtw_m.warping_paths(x, y)[0] # barely possible
    assert np.isfinite(d)
    
    y[3] = np.nan
    
    d = dtw_m.warping_paths(x, y)[0] # impossible
    assert np.isinf(d)
    
    d = dtw_m.warping_paths(x, y, missing_value_restrictions='partial')[0] # possible thanks to the relaxed restrictions
    assert np.isfinite(d)


def test_dtw_arow_multivariate(D=2):
    # Test DTW-AROW on multivariate time series that have the same distance as their univariate counterparts:
    x = np.array([1, 1, 2, 2, 10, 1, 1], dtype=np.float32)
    y = np.array([1, 1, 1, 2, 2, 13, 13, 1, 1, 1], dtype=np.float32)
    y[2:4] = np.nan
    d_exp = np.sqrt(2 * (3 ** 2))
    
    x, y = convert_univariate_into_multivariate(x, y, D)
    
    d = dtw_m.warping_paths(x, y, missing_value_adjustment=None)[0]
    assert d == pytest.approx(d_exp)
    
    d = dtw_m.warping_paths(x, y)[0]
    d_exp = d_exp * np.sqrt((len(x) + len(y)) / (np.sum(~np.any(np.isnan(x), axis=1)) + np.sum(~np.any(np.isnan(y), axis=1))))
    assert d == pytest.approx(d_exp)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    test_dtw()
    test_dtw_arow()
    test_dtw_with_partialrestrictions()
    test_consistency_of_dtw_and_dtw_missing()
    test_impossible_warping_in_dtw_arow_and_dtw_with_partialrestrictions()
    test_dtw_arow_multivariate(2)
    test_dtw_arow_multivariate(3)
    test_dtw_arow_multivariate(10)