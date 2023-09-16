import numpy as np
import logging
from dtaidistance import innerdistance
from dtaidistance.exceptions import NumpyException
import dtaidistance.dtw as dtaidistance_dtw

import dtw_missing.missing_utils as mus

logger = logging.getLogger("be.kuleuven.dtw_missing")

inf = float("inf")

argmin = np.argmin
argmax = np.argmax

# Function to test whether time samples of a time series are considered as missing or not:
# missing_fun_univ = np.isnan(s) # univariate
# missing_fun_multiv = lambda s: np.isnan(s).any(axis=1) # multivariate (the first dimension is time)
MISSING_FUN_DEFAULT = lambda s: np.isnan(s).any(axis=tuple(range(1, s.ndim))) # the first dimension is time

COST_OF_MISSING = 0 # cost of a comparison between two time instants that involves any missing values

def warping_paths(s1, s2, window=None, max_dist=None, use_pruning=False,
                  max_step=None, max_length_diff=None, penalty=None, psi=None, psi_neg=True,
                  use_c=False, use_ndim=False, inner_dist=innerdistance.default,
                  cost_matrix=None, 
                  result_fn=None, 
                  missing_value_restrictions="full", 
                  missing_value_adjustment="proportion_of_missing_values", 
                  missing_fun=None, 
                  return_optimal_warping_path=False):
    """
    Dynamic Time Warping (DTW) that can handle missing values.
    
    This is called DTW-AROW (DTW with Additional Restrictions On Warping) in 
        Aras Yurtman, Jonas Soenen, Wannes Meert, Hendrik Blockeel, 
        "Estimating Dynamic Time Warping Distance Between Time Series with Missing Data," 
        European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 
        (ECML-PKDD) 2023. https://lirias.kuleuven.be/4091041?limo=0
    
    Built on the DTAIDistance implementation: 
    https://github.com/wannesm/dtaidistance/blob/master/dtaidistance/dtw.py

    The full matrix of all warping paths (or accumulated cost matrix) is built.
    
    Returns the DTW(-AROW) distance, the accumulated cost matrix, and optionally the optimal warping path.

    #TODO: explanations of the input arguments
    
    #TODO: check the number of output arguments for early "return inf" statements
    
    :param s1: First sequence
    :param s2: Second sequence
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param use_pruning: Not supported, kept for compatibility with dtaidistance.dtw.warping_paths.
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :param psi_neg: Replace values that should be skipped because of psi-relaxation with -1.
    :param use_c: Use the C implementation instead of Python
    :param use_ndim: The input series is >1 dimensions.
        Use cost = SquaredEuclideanDistance(s1[i], s2[j])
    :param inner_dist: Distance between two points in the time series.
        One of 'squared euclidean' (default), 'euclidean'
    :param cost_matrix: Precomputed cost matrix. 
                        If specified, inner_dist is overridden; otherwise, inner_dist is used.
    :param result_fn: The function that is applied on the cumulative cost on the optimal warping path. 
                      If specified, inner_dist is overridden; otherwise, inner_dist is used.
    :param missing_value_restrictions: restrictions on warping in case of missing values (DEFAULT: "full")
        "full"      : DTW-AROW technique: Restrict the warping path to be diagonal 
                      so that every missing interval will be matched with an interval of the same length form the other time series.
        "partial"   : Prevent the warping path from being horizontal if the row corresponds to missing values, and
                                               from being vertical if the column corresponds to misisng values, 
                      so that every missing interval will be matched with an equal-length or shorter interval from the other time series.
        "none", None: Do not restrict the warping path. Missing values might cause undesired warpings and an underestimated DTW distance.
    :param missing_value_adjustment: Adjust the DTW distance to compensate for missing values that are matched at zero cost. (DEFAULT: "proportion_of_missing_values")
        "proportion_of_missing_values"    : Scale up the DTW distance according to the proportion of 
                                            total missing values in the two time seriees.
        "proportion_of_missing_comparisons": Scale up the DTW distance according to the proportion of 
                                             the points (on the optimal warping path) that corresponds to missing values.
        "none" or None                    : Do not scale up the DTW distance. Missing values might cause the distance to be underestimated.
    :param missing_fun: function (applied on s1 and s2) to check whether each time instant is missing or not
                        (DEFAULT: None: consider a time instant as missing if any dimension has a missing value np.nan)
    :param return_optimal_warping_path: Return the optimal warping path.
    :returns: (DTW distance, DTW matrix)                       if return_optimal_warping_path==False, 
              (DTW distance, DTW matrix, optimal warping path) if return_optimal_warping_path==True.
    """
    if use_c:
        msg = "C implementation for DTW-AROW is not available"
        logger.warning(msg)
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")
    
    # Always use ndim to use np functions
    cost, result_fn_innerdistance = innerdistance.inner_dist_fns(inner_dist, use_ndim=True) # cost is not used if cost_matrix is specified, and result_fn_innerdistance is not used if it is specified
    if result_fn is None:
        result_fn = result_fn_innerdistance
    
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        if return_optimal_warping_path:
            return inf, None, None
        else:
            return inf, None
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = inf
    else:
        max_step *= max_step
    if use_pruning:
        # max_dist = dtaidistance_dtw.ub_euclidean(s1, s2)**2
        msg = "This type of pruning is not supported for DTW-AROW."
        logger.warning(msg)
    elif not max_dist:
        max_dist = inf
    else:
        max_dist *= max_dist
    if penalty is None:
        penalty = 0
    else:
        penalty *= penalty
    psi_1b, psi_1e, psi_2b, psi_2e = dtaidistance_dtw._process_psi_arg(psi)
    
    if missing_value_restrictions is None or missing_value_restrictions == "none":
        missing_restrict = False
    elif missing_value_restrictions == "full":
        missing_restrict = True
        missing_restrict_partial = False
    elif missing_value_restrictions == "partial":
        missing_restrict = True
        missing_restrict_partial = True
    else:
        msg = "missing_restrict must be None, 'none', 'full', or 'partial'."
        logger.error(msg)
        raise Exception(msg)
    
    if missing_fun is None:
        missing_fun = MISSING_FUN_DEFAULT
    
    s1_isavailable = ~missing_fun(s1)
    s2_isavailable = ~missing_fun(s2)
    
    if psi is None and cost_matrix is None and missing_value_restrictions in ['full', 'partial']:
        custom_left_right_bounds, constrained_warping_path_possible = calculate_missing_bounds_fast(s1, s2, missing_fun, missing_value_restrictions)
        if not constrained_warping_path_possible:
            if return_optimal_warping_path:
                return inf, None, None
            else:
                return inf, None
    else:
        custom_left_right_bounds = None
    
    path = None
    
    if return_optimal_warping_path or missing_value_adjustment=="proportion_of_missing_comparisons":
        calculate_optimal_warping_path = True
    else:
        calculate_optimal_warping_path = False
    
    if calculate_optimal_warping_path:
        steps = np.full((r, c), np.nan) # 0: diagonal, 1: vertical, 2: horizontal
    
    dtw = np.full((r + 1, c + 1), inf)
    # dtw[0, 0] = 0
    for i in range(psi_2b + 1):
        dtw[0, i] = 0
    for i in range(psi_1b + 1):
        dtw[i, 0] = 0
    i0 = 1
    i1 = 0
    sc = 0
    ec = 0
    smaller_found = False
    ec_next = 0
    for i in range(r):
        i0 = i
        i1 = i + 1
        if custom_left_right_bounds is None:
            j_start = max(0, i - max(0, r - c) - window + 1)
            j_end = min(c, i + max(0, c - r) + window)
        else:
            j_start = custom_left_right_bounds[0][i]
            j_end = custom_left_right_bounds[1][i] + 1 # "+1" because the range function does not include the right bound
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        # jmin = max(0, i - max(0, r - c) - window + 1)
        # jmax = min(c, i + max(0, c - r) + window)
        # print(i,jmin,jmax)
        # x = dtw[i, jmin-skipp:jmax-skipp]
        # y = dtw[i, jmin+1-skipp:jmax+1-skipp]
        # print(x,y,dtw[i+1, jmin+1-skip:jmax+1-skip])
        # dtw[i+1, jmin+1-skip:jmax+1-skip] = np.minimum(x,
        #                                                y)
        for j in range(j_start, j_end):
            # print('j =', j, 'max=',min(c, c - r + i + window))
            if cost_matrix is None:
                if s1_isavailable[i] and s2_isavailable[j]:
                    d = cost(s1[i], s2[j])
                else:
                    d = COST_OF_MISSING
            else:
                d = cost_matrix[i, j]
            if max_step is not None and d > max_step:
                continue
            # print(i, j + 1 - skip, j - skipp, j + 1 - skipp, j - skip)
            
            if missing_restrict and s1_isavailable[i0] and (missing_restrict_partial or (s2_isavailable[max(0, j-1)] and s2_isavailable[j])):
                penalty_horizontal = 0
            else:
                penalty_horizontal = np.inf
            if missing_restrict and s2_isavailable[j] and (missing_restrict_partial or (s1_isavailable[max(0, i0-1)] and s1_isavailable[i0])):
                penalty_vertical = 0
            else:
                penalty_vertical = np.inf
                
            dtw[i1, j + 1] = d + min(dtw[i0, j],
                                     dtw[i0, j + 1] + penalty + penalty_vertical,
                                     dtw[i1, j] + penalty + penalty_horizontal)
            
            if calculate_optimal_warping_path:
                steps[i, j] = np.argmin([dtw[i0, j],
                                         dtw[i0, j + 1] + penalty + penalty_vertical,
                                         dtw[i1, j] + penalty + penalty_horizontal])
            
            if dtw[i1, j + 1] > max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
    
    # Decide which d to return
    dtw = result_fn(dtw)
    if psi_1e == 0 and psi_2e == 0:
        d_position = i1, min(c, c + window - 1)
    else:
        ir = i1
        ic = min(c, c + window - 1)
        if psi_1e != 0:
            vr = dtw[ir:max(0, ir-psi_1e-1):-1, ic] # bottom part of the last column
            mir = argmin(vr)
            vr_mir = vr[mir]
            d_position_r = ir - mir, ic
        else:
            mir = ir
            vr_mir = inf
            d_position_r = None
        if psi_2e != 0:
            vc = dtw[ir, ic:max(0, ic-psi_2e-1):-1] # right part of the bottom row
            mic = argmin(vc)
            vc_mic = vc[mic]
            d_position_c = ir, ic - mic
        else:
            mic = ic
            vc_mic = inf
            d_position = None
        if vr_mir < vc_mic:
            if psi_neg:
                dtw[ir:ir-mir:-1, ic] = -1
            d_position = d_position_r
            # d = vr_mir
            # assert dtw[d_position] == d
        else:
            if psi_neg:
                dtw[ir, ic:ic-mic:-1] = -1
            d_position = d_position_c
            # d = vc_mic
            # assert dtw[d_position] == d
    
    if d_position is None:
        d = None
    else:
        d = dtw[d_position]
    
    if calculate_optimal_warping_path:
        if d_position is None:
            path = None
        else:
            i, j = d_position[0] - 1, d_position[1] - 1
            path = [(i, j)]
            while i >= 0 and j >= 0:
                if np.isnan(steps[i, j]): # undefined
                    break
                elif steps[i, j] == 0: # diagonal
                    i -= 1
                    j -= 1
                elif steps[i, j] == 1: # vertical
                    i -= 1
                elif steps[i, j] == 2: # horizontal
                    j -= 1
                if dtw[i, j] != -1:
                    path.append((i, j))
            path.pop()
            path.reverse()

    d = d*calculate_adjustment_factor(s1, s2, 
                                      missing_value_adjustment = missing_value_adjustment, 
                                      path = path)
    
    if max_dist and d*d > max_dist:
        d = inf
        path = None
    
    if return_optimal_warping_path:
        return d, dtw, path
    else:
        return d, dtw


def count_missing(s, missing_fun=None): # count the rows with any missing (nan) values in an array
    if missing_fun is None:
        missing_fun = MISSING_FUN_DEFAULT
    return sum(missing_fun(s))


def calculate_adjustment_factor(s1, s2, missing_value_adjustment, path=None, missing_fun=None):
    """Calculate the adjustment factor (by which the DTW distance should be multiplied).

    :param s1: First sequence.
    :param s2: Second sequence.
    :param missing_value_adjustment: Type of adjustment:
        None or "none"                    : No adjustment.
        "proportion_of_missing_values"    : Adjustment according to the proportion of non-missing time samples (averaged over the two time series).
        "avg_nonmissing_warping_length"   : Adjustment according to the average of the minimum and maximum possible length of the warping path.
        "proportion_of_missing_comparisons": Adjustment according to the proportion of points (on the optimal warping path) that correspond to non-missing comparisons.
    :param path: Optimal warping path that is used when missing_value_adjustment is "proportion_of_missing_comparisons". DEFAULT: None.
    :param missing_fun: function (applied on s1 and s2) to check whether each time instant is missing or not
                        (DEFAULT: None: consider a time instant as missing if any dimension has a missing value np.nan)
    :return: Adjustment factor (by which the DTW distance should be multiplied).
    """
    
    if missing_fun is None:
        missing_fun = MISSING_FUN_DEFAULT
    
    M1 = len(s1)
    M2 = len(s2)
    M1_nonmiss = M1 - count_missing(s1)
    M2_nonmiss = M2 - count_missing(s2)
    
    if missing_value_adjustment is None or missing_value_adjustment == 'none':
        adjustment_factor = 1
        
    elif missing_value_adjustment == 'proportion_of_missing_values': # adjust by the number of non-missing time samples (averaged over the two time series)
        adjustment_factor = (M1 + M2)/(M1_nonmiss + M2_nonmiss)
        adjustment_factor = np.sqrt(adjustment_factor) # to take into account the square root in DTW
        
    elif missing_value_adjustment == 'avg_nonmissing_warping_length': # adjust by the average of the minimum and maximum possible length of the warping path
        avg_warping_path_length_nonmiss = (max(M1, M2) + (M1 + M2))/2
        avg_warping_path_length = (max(M1_nonmiss, M2_nonmiss) + (M1_nonmiss + M2_nonmiss))/2
        adjustment_factor = avg_warping_path_length_nonmiss/avg_warping_path_length # DTW distance will be MULTIPLIED by this
        adjustment_factor = np.sqrt(adjustment_factor) # to take into account the square root in DTW
        
    elif missing_value_adjustment == 'proportion_of_missing_comparisons': # calculate the adjustment factor from path (the optimal warping path).
        if path is None:
            msg = "path must be specified when missing_value_adjustment is 'proportion_of_missing_comparisons'."
            logger.error(msg)
            raise Exception(msg)
        s1_missing = missing_fun(s1)
        s2_missing = missing_fun(s2)
        path_nonmissing = [not (s1_missing[p[0]] or s2_missing[p[1]]) for p in path] # points that do NOT involve missing values in path (the optimal warping path)
        if np.sum(path_nonmissing) == 0: # all matches involve missing values
            adjustment_factor = np.sqrt(len(path_nonmissing)) # = np.sqrt(1/(1/len(path_nonmissing))) # calculate it by assuming that only one match is non-missing in the warping path, although adjustment_factor does not matter in this case because the total cost and the DTW distance will be zero
        else:
            adjustment_factor = np.sqrt(1/np.mean(path_nonmissing)) # square root of the ratio between the total number of matches in the warping path (i.e., the path length) and the number of matches that do NOT involve missing values
            
    else:
        msg = "missing_value_adjustment must be "
        logger.error(msg)
        raise Exception(msg)
    
    return adjustment_factor


def calculate_missing_bounds_fast(s1, s2, missing_fun=None, missing_value_restrictions="full"): 
    """ 
    Calculate left and right bounds (for every row) outside which the elements of the cumulative cost matrix 
    don't contribute to the DTW distance under the constraints imposed in DTW-AROW, 
    either for the standard DTW-AROW (missing_value_restrictions=='full') 
    or its "relaxed" version with partial restrictions on warping (missing_value_restrictions='partial').
    Also return whether a warping path is possible or not under these constraints.

    :param s1: first time series
    :param s2: second time series
    :param missing_fun: function (applied on s1 and s2) to check whether each time instant is missing or not
                        (DEFAULT: None: consider a time instant as missing if any dimension has a missing value np.nan)
    :param missing_value_restrictions: restrictions on warping in case of missing values (DEFAULT: "full")
        "full"      : DTW-AROW technique: Restrict the warping path to be diagonal 
                      so that every missing interval will be matched with an interval of the same length form the other time series.
        "partial"   : Prevent the warping path from being horizontal if the row corresponds to missing values, and
                                               from being vertical if the column corresponds to misisng values, 
                      so that every missing interval will be matched with an equal-length or shorter interval from the other time series.
    :return: [left bound, right bound], whether constrained warping path is possible or not
    """    
    if missing_fun is None:
        missing_fun = MISSING_FUN_DEFAULT
    
    if missing_value_restrictions=='full':
        missing_restrict_partial = False
    elif missing_value_restrictions=='partial':
        missing_restrict_partial = True
    else:
        msg = "missing_value_restrictions must be either 'full' or 'partial'."
        logger.error(msg)
        raise Exception(msg)
       
    s1_missing = missing_fun(s1)
    s2_missing = missing_fun(s2)
    
    r = len(s1)
    c = len(s2)
    
    # Calculate the left bound:
    
    # # SAME RESULT:
    # constrained_warping_path_possible = True # if False, then "impossible path"
    # 
    # leb1 = [0 for i in range(r)] # left bound 1 for the path starting from the top left
    # j = 0 # column index
    # rightmost_column_exceeded = False
    # for i in range(0, r-1): # row
    #     if rightmost_column_exceeded:
    #         leb1[i+1] = c-1
    #     else:
    #         if s1_missing[i] or s1_missing[min(i+1, r-1)] or s2_missing[j]: # if missing
    #             j += 1 # diagonal
    #         if j == c:
    #             rightmost_column_exceeded = True
    #             constrained_warping_path_possible = False
    #             j -= 1
    #         leb1[i+1] = j
    # 
    # leb2 = [c-1 for i in range(r)] # left bound 2 for the path starting from the bottom right
    # i = r-1 # row index
    # j = c-1 # column index
    # leftmost_column_reached = False
    # while i >= 0 and j >= 0: # row (reversed)
    #     leb2[i] = min(leb2[i], j)
    #     if leftmost_column_reached:
    #         i -= 1
    #         # leb_2[i] = 0
    #     else:
    #         if s1_missing[i] or s2_missing[j-1] or s2_missing[j]: # if missing
    #             i -= 1 # \
    #             j -= 1 # / diagonal
    #         else:
    #             j -= 1 # left
    #         if j == 0:
    #             leftmost_column_reached = True
    # if i < 0:
    #     constrained_warping_path_possible = False
    # leb = [max(v1, v2) for v1, v2 in zip(leb1, leb2)]
    
    constrained_warping_path_possible = True # if False, then "impossible path"
    
    leb1 = [c-1 for i in range(r)] # left bound 1 for the path starting from the top left
    j = 0 # column index
    leb1_impossible = False
    for i in range(0, r): # row
        leb1[i] = j
        if (not missing_restrict_partial and (s1_missing[i] or s1_missing[min(i+1, r-1)])) or s2_missing[j]: # if missing_restrict_partial=True, then check whether s2 is missing or not; otherwise, check if any of s1 or s2 is missing
            j += 1 # diagonal
        if j > c-1:
            if i < r-1:
                leb1_impossible = True # last row not reached
            break # rightmost column exceeded
    
    leb2 = [0 for i in range(r)] # left bound 2 for the path starting from the bottom right
    i = r-1 # row index
    leb2_impossible = False
    for j in range(c-1, -1, -1): # column (reversed)
        leb2[i] = j
        if s1_missing[i] or (not missing_restrict_partial and (s2_missing[max(j-1, 0)] or s2_missing[j])): # if missing_restrict_partial=True, then check whether s1 is missing or not; otherwise, check if any of s1 or s2 is missing
            i -= 1 # diagonal
        if i < 0:
            if j > 0:
                leb2_impossible = True # first column not reached
            break # first row exceeded
    leb = [max(v1, v2) for v1, v2 in zip(leb1, leb2)]
    
    constrained_warping_path_possible = not (leb1_impossible or leb2_impossible) # warping path is not possible if any of leb1 and leb2 are impossible
    
    # Calculate the right bound:
    
    rib1 = [c-1 for i in range(r)] # right bound 1 for the path starting from top left
    i = 0 # row index
    for j in range(0, c): # column
        rib1[i] = j
        if s1_missing[i] or (not missing_restrict_partial and (s2_missing[j] or s2_missing[min(j+1, c-1)])): # if missing_restrict_partial=True, then check whether s1 is missing or not; otherwise, check if any of s1 or s2 is missing
            i += 1 # diagonal
        if i > r-1:
            break # last row exceeded
    
    rib2 = [0 for i in range(r)] # right bound 2 for the path starting from bottom right
    j = c-1 # column index
    for i in range(r-1, -1, -1): # row (reversed)
        rib2[i] = j
        if (not missing_restrict_partial and (s1_missing[i] or s1_missing[max(i-1, 0)])) or s2_missing[j]: # if missing_restrict_partial=True, then check whether s2 is missing or not; otherwise, check if any of s1 or s2 is missing
            j -= 1 # diagonal
        if j < 0:
            break # leftmost column exceeded
    rib = [min(v1, v2) for v1, v2 in zip(rib1, rib2)]
    # rib = [c-1 for i in range(r)] # dummy result
    
    return [leb, rib], constrained_warping_path_possible