# -*- coding: UTF-8 -*-
"""
a generalized version of
Dynamic Time Warping Barycenter Averaging (DBA) 
that can handle missing values using DTW-AROW

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Based on the dtaidistance.dtw_barycenter: 
    https://github.com/wannesm/dtaidistance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import logging
import math
import array
import random
import copy

import numpy as np

from dtaidistance import warping_path, distance_matrix
# from . import ed
# from . import util
# from . import util_numpy
from dtaidistance.util import SeriesContainer
from dtaidistance.exceptions import NumpyException

import dtw_missing.dtw_missing as dtw_m
import dtw_missing.missing_utils as mus
from dtw_missing import dtw_barycenter_imputation

import warnings


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_cc = None


def get_good_c(s, mask, nb_initial_samples, use_c=False, **kwargs):
    if nb_initial_samples > len(s):
        nb_initial_samples = len(s)
    mask_size = np.sum(mask)
    cs = []
    randthr = nb_initial_samples / mask_size
    for i in range(len(s)):
        if mask[i]:
            if random.random() <= randthr:
                cs.append(s[i])
        if len(cs) == nb_initial_samples:
            break
        else:
            randthr = (nb_initial_samples - len(cs)) / (mask_size - i - 1)
    d = distance_matrix(cs, use_c=use_c,  **kwargs)
    d = d.sum(axis=1)
    best_i = np.argmin(d)
    return s[best_i]


def dba_loop(s, c=None, max_it=10, thr=0.001, mask=None,
             keep_averages=False, use_c=False, nb_initial_samples=None, nb_prob_samples=None, 
             use_dtw_missing=False, iterative_imputation=False,
             **kwargs):
    """Loop around the DTW Barycenter Averaging (DBA) method until convergence.

    :param s: Container of sequences
    :param c: Initial averaging sequence.
        If none is given, the first one is used (unless if nb_initial_samples is set).
        Better performance can be achieved by starting from an informed
        starting point (Petitjean et al. 2011).
    :param max_it: Maximal number of calls to DBA.
    :param thr: Convergence if the DBA is changing less than this value.
    :param mask: Boolean array with the series in s to use. If None, use all.
    :param keep_averages: Keep all DBA values (for visualisation or debugging).
    :param nb_initial_samples: If c is None, and this argument is not None, select
        nb_initial_samples samples and select the series closest to all other samples
        as c.
    :param nb_prob_samples: Probabilistically sample the best path instead of the
        deterministic version.
    :param use_c: Use a fast C implementation instead of a Python version.
    :param use_dtw_missing: Use the version of DTW that can handle missing values (with parameters provided in kwargs) (no C implementation).
    :param iterative_imputation: In each iteration, impute the time samples (that were missing in the beginning) using the DBA (no C implementation).
    :param kwargs: Arguments for dtw.distance
    """
    if np is None:
        raise NumpyException('The method dba_loop requires Numpy to be available')
    s = SeriesContainer.wrap(s)
    avg = None
    avgs = None
    if keep_averages:
        avgs = []
    if mask is None:
        mask = np.full((len(s),), True, dtype=bool)
    if nb_prob_samples is None:
        nb_prob_samples = 0
    if c is None:
        if nb_initial_samples is None:
            curi = 0
            while mask[curi] is False:
                curi += 1
            c = s[curi]
        else:
            c = get_good_c(s, mask, nb_initial_samples, use_c=use_c, **kwargs)

        # You can also use a constant function, but this gives worse performance.
        # After the first iteration, this will be the average of all
        # sequences. The disadvantage is that this might create e.g. multiple
        # peaks for a sequence with only one peak (but shifted) and then the
        # original sequences will map their single peak to the different peaks
        # in the first average and converge to that as a local optimum.
        # t = s.get_avg_length()
        # c = array.array('d', [0] * t)
    if use_c:
        if np is not None and isinstance(mask, np.ndarray):
            # The C code requires a bit array of uint8 (or unsigned char)
            mask_copy = np.packbits(mask, bitorder='little')
        else:
            raise Exception('Mask only implemented for C when passing a Numpy array. '
                            f'Got {type(mask)}')
    else:
        mask_copy = mask
    if use_dtw_missing and use_c:
        warnings.warn('The DTW version that can handle missing values is not implemented in C! Continuing with Python implementation...')
        use_c = False
    if iterative_imputation and use_c:
        warnings.warn('Iterative imputation not implemented in C! Continuing with Python implementation...')
        use_c = False
    if not use_c and nb_prob_samples != 0:
        warnings.warn('The parameter nb_prob_samples is not available in the Python implementation! Continuing with nb_prob_samples = 0...')

    if iterative_imputation:
        s_orig = copy.deepcopy(s)
        s_orig_ismissing = list(map(np.isnan, s_orig))
            

    for it in range(max_it):
        logger.debug(f'DBA Iteration {it}')
        if use_c:
            assert(c is not None)
            c_copy = c.copy()  # The C code reuses this array
            dtw_cc.dba(s, c_copy, mask=mask_copy, nb_prob_samples=nb_prob_samples, **kwargs)
            avg = c_copy
        else:
            if not use_c:
                # if nb_prob_samples != 0:
                #     warnings.warn('The parameter nb_prob_samples is not available in the Python implementation! Continuing with nb_prob_samples = 0...')
                # avg = dba(s, c, mask=mask, nb_prob_samples=nb_prob_samples, use_c=use_c, **kwargs) # nb_prob_samples not available in the Python implementation of dba
                avg = dba(s, c, mask=mask, use_c=use_c, use_dtw_missing=use_dtw_missing, **kwargs)
                
                if use_dtw_missing:
                    avg = mus.interpolate_missing(avg) # interpolate missing parts that may remain because of the lack of matching to any non-missing samples [DIFFERENT FROM PREVIOUS IMPLEMENTATION]
                
                if iterative_imputation:
                    s_withmissing = copy.deepcopy(s)
                    for i in range(len(s_withmissing)):
                        s_withmissing[i][s_orig_ismissing[i]] = np.nan # set the time samples that were originally missing to np.nan
                    s = dtw_barycenter_imputation.impute_with_dba_multiple(s_withmissing, avg, **kwargs)
                
            else:
                avg = dba(s, c, mask=mask, nb_prob_samples=nb_prob_samples, use_c=use_c, **kwargs)
        if keep_averages:
            avgs.append(avg)
        if thr is not None and c is not None:
            diff = 0
            # diff = np.sum(np.subtract(avg, c))
            for av, cv in zip(avg, c):
                diff += abs(av - cv)
            diff /= len(avg)
            if diff <= thr:
                logger.debug(f'DBA converged at {it} iterations (avg diff={diff}).')
                break
        c = avg
    
    out = [avg]
    if keep_averages:
        out.append(avgs)
    if iterative_imputation:
        out.append(s)
    return tuple(out)


def dba(s, c, mask=None, samples=None, use_c=False, nb_initial_samples=None, use_dtw_missing=False, **kwargs):
    """DTW Barycenter Averaging.

    F. Petitjean, A. Ketterlin, and P. Gan ̧carski.
    A global averaging method for dynamic time warping, with applications to clustering.
    Pattern Recognition, 44(3):678–693, 2011.

    :param s: Container of sequences
    :param c: Initial averaging sequence.
        If none is given, the first one is used (unless if nb_initial_samples is set).
        Better performance can be achieved by starting from an informed
        starting point (Petitjean et al. 2011).
    :param mask: Boolean array with the series in s to use. If None, use all.
    :param nb_initial_samples: If c is None, and this argument is not None, select
        nb_initial_samples samples and select the series closest to all other samples
        as c.
    :param use_c: Use a fast C implementation instead of a Python version.
    :param use_dtw_missing: Use the version of DTW that can handle missing values (with parameters provided in kwargs).
    :param kwargs: Arguments for dtw.distance
    :return: Bary-center of length len(c).
    """
    if samples is not None and samples > 0:
        # TODO
        raise Exception("Prob sampling for DBA not yet implemented for Python")
    if use_dtw_missing and use_c:
        warnings.warn('The DTW version that can handle missing values is not implemented in C! Continuing with Python implementation...')
        use_c = False
        
    s = SeriesContainer.wrap(s)
    if mask is not None and not mask.any():
        # Mask has not selected any series
        print("Empty mask, returning zero-constant average")
        c = array.array('d', [0] * len(s[0]))
        return c
    if mask is None:
        mask = np.full((len(s),), True, dtype=bool)
    if c is None:
        if nb_initial_samples is None:
            curi = 0
            while mask[curi] is False:
                curi += 1
            c = s[curi]
        else:
            c = get_good_c(s, mask, nb_initial_samples, use_c=use_c, **kwargs)
    t = len(c)
    assoctab = [[] for _ in range(t)]
    for idx, seq in enumerate(s):
        if mask is not None and not mask[idx]:
            continue
        if use_c:
            m = dtw_cc.warping_path(c, seq, **kwargs)
        else:
            m = warping_path(c, seq, use_dtw_missing=use_dtw_missing, **kwargs)
        for i, j in m:
            assoctab[i].append(seq[j])
    cp = array.array('d', [0] * t)
    for i, values in enumerate(assoctab):
        if len(values) == 0:
            print('WARNING: zero values in assoctab')
            print(c)
            for seq in s:
                print(seq)
            print(assoctab)
        
        cp[i] = np.nanmean(values)
        # cp[i] = sum(values) / len(values)  # barycenter
        
    if np.isnan(values).any():
        cp = array.array(cp.typecode, mus.interpolate_missing(np.array(cp))) # imterpolate if the DBA has missing values (which happens when, for any time sample, all of the values associated with it are missing)
    
    return cp
