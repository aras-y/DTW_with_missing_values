import numpy as np
import logging
import copy

import dtw_missing.dtw_missing as dtw_m
import dtw_missing.missing_utils as mus
from dtw_missing import dtw_barycenter_imputation

from tools.parallel_cdist import cdist_generic
from dtaidistance import dtw_barycenter
from sklearn_extra.cluster import KMedoids
from scipy.stats import mode
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger("be.kuleuven.dtw_missing")


def dtw_arow_distance(s1, s2, dtw_params={}):
    """
    Compute the DTW-AROW (or variants of DTW-AROW) distance between two time series.
    
    :param s1: First sequence.
    :param s2: Second sequence.
    :param dtw_params: Parameters to pass to dtw_missing.warping_paths(). DEFAULT: {}.
    :return: DTW(-AROW) distance between s1 and s2.
    """    
    return dtw_m.warping_paths(s1, s2, **dtw_params)[0]


def dtw_arow_distance_matrix(dataset, dtw_params={}, n_jobs=-1, progress_bar=False):
    """
    Compute pairwise DTW-AROW (or other variants of DTW-AROW) distances in a dataset.
    
    :param dataset: Dataset. An iterable, or a 2-D or 3-D numpy array.
                    A fixed-length univariate or multivariate dataset can be stored as a 2-D or 3-D array, respectively, 
                        or as an iterable that contains 1-D or 2-D numpy arrays, respectively. 
                    A variable-length univariate or multivariate dataset has to be an iterable 
                        that contains 1-D or 2-D numpy arrays, respectively. 
                        1-D arrays can have different lengths, whereas 
                        2-D arrays can have different number of columns but has to have same number of rows.
    :param dtw_params: Parameters to pass to dtw_missing.warping_paths(). DEFAULT: {}.
    :param n_jobs: The number of parallel jobs. DEFAULT: -1 (use all available resources).
    :param progress_bar: Whether to show a progress bar or not. DEFAULT: False.
    :return: Pairwise distance matrix.
    """
    
    return cdist_generic(
                         #dist_fun=dtw_arow_distance, 
                         dist_fun=lambda s1, s2: dtw_arow_distance(s1, s2, dtw_params),
                         dataset1=dataset, 
                         n_jobs=n_jobs, 
                         progress_bar=progress_bar, 
                         compute_diagonal=False, 
                         parallel_method='processes',
                         #**dtw_params
                        )


DEFAULT_DTW_CAI_PARAMS = dict(iterative_imputation=False, 
                              no_clusters='elbow',
                              maxiter_kmeans=100,
                              maxiter_dba=1, 
                              no_clusters_range_for_elbow=(2,15), 
                              dba_init='medoid',
                              random_state=None,
                              dba_thr=0,
                             )


class Experiment:
    """
    generic experiment class
    """
    
    def __init__(self):
        self._dataset = None
    
    
    @property
    def dataset(self):
        return self._dataset
    
    
    @dataset.setter
    def dataset(self, dataset):
        """
        Set dataset.

        :param dataset: Dataset. An iterable, or a 2-D or 3-D numpy array.
                A fixed-length univariate or multivariate dataset can be stored as a 2-D or 3-D array, respectively, 
                    or as an iterable that contains 1-D or 2-D numpy arrays, respectively. 
                A variable-length univariate or multivariate dataset has to be an iterable 
                    that contains 1-D or 2-D numpy arrays, respectively. 
                    1-D arrays can have different lengths, whereas 
                    2-D arrays can have different number of columns but has to have same number of rows.
        """
        
        self._dataset = dataset
    
    
    def preprocess_dataset(self, dataset_preproc_params=None):
        #TODO
        pass
    
    
    def compute_pairwise_distances(self, missing_method, missing_method_params={}, n_jobs=-1, progress_bar=False):
        """
        Compute pairwise DTW-AROW or DTW-CAI distances (or their variants).

        :param missing_method: Method to handle missing values: 
                               'dtw_arow': DTW-AROW (or its variants)
                               'dtw_cai': DTW-CAI (or its variants)
        :param missing_method_params: Parameters for the method that handles missing values. 
                                      For DTW-AROW, dtw_params that are passed to dtw_missing.warping_paths().
                                      For DTW-CAI, (dtw_params, dtw_cai_params). 
                                          dtw_cai_params: 
                                              iterative_imputation: Whether to impute missing values in every iteration or not. 
                                                                    DEFAULT: False.
                                              no_clusters: Number of clusters for k-means or 'elbow' to compute it automatically 
                                                           using the elbow method. 
                                              maxiter_kmeans: Maximum number of iterations for k-means (0 for unlimited).
                                                              DEFAULT: 100
                                              maxiter_dba: Maximum number of iterations to execute DBA (in every iteration of k-means).
                                                           0: Instead of DBA, use medoid with linear interpolation. 
                                                           -1: Instead of DBA, use mean with linear interpolation.
                                                           DEFAULT: 1 (only a single iteration of DBA)
                                              no_clusters_range_for_elbow: For the elbow method, the range (as a tuple) for the number of clusters.
                                                                           DEFAULT: (2,15)
                                              dba_init: Initialization for DBA.
                                                        None:     random initialization
                                                        'medoid': initialize with the medoid
                                              random_state: Random seed for DBA initialization (used when dba_init is None).
                                              dba_thr: Threshold on the change in the barycenter average (as the mean absolute difference) 
                                                       used as a stopping criterion for DBA.
        :param n_jobs: The number of parallel jobs. DEFAULT: -1 (use all available resources).
        :param progress_bar: Whether to show a progress bar or not. DEFAULT: False.
        :return: Pairwise distance matrix D.
        """        
        if missing_method == 'dtw_arow':
            self.D = dtw_arow_distance_matrix(self._dataset, missing_method_params, n_jobs, progress_bar)
        elif missing_method == 'dtw_cai':
            self.__dtw_cai(missing_method_params, n_jobs, progress_bar)
        
        return self.D
    
    
    def get_pairwise_distances(self):
        return self.D
    
    
    def __dtw_cai(self, missing_method_params, n_jobs, progress_bar):
        # Execute the DTW-CAI algorithm (or its variants).
        
        show_elbow = False
        
        save_imputed_all_iterations = True
        save_additional_clustering_results = True
        
        if missing_method_params == {}:
            missing_method_params = [{}, {}]
        dtw_params = missing_method_params[0]
        dtw_cai_params = missing_method_params[1]
        
        dtw_cai_params = DEFAULT_DTW_CAI_PARAMS | dtw_cai_params # set default values for unspecified parameters
        
        iterative_imputation = dtw_cai_params['iterative_imputation'] # whether to impute missing values in every iteration or not
        k_iclustering = dtw_cai_params['no_clusters'] # k of k-means for iterative clustering or 'elbow' for the elbow method
        maxiter_kmeans = dtw_cai_params['maxiter_kmeans'] # maximum number of iterations for k-means (0 for unlimited)
        maxiter_dba = dtw_cai_params['maxiter_dba'] # maximum number of iterations for DBA in each k-means iteration (0: medoid with linear interpolation, -1: mean with linear interpolation)
        if 'missing_value_restictions' in dtw_params and dtw_params['missing_value_restictions'] == 'partial':
            # dtw_arow_partial_iclustering = True # relaxed version of DTW-AROW with partial restrictions on warping
            missing_method_partial_dba = True # relaxed version of DTW-AROW with partial restrictions on warping
        else:
            # dtw_arow_partial_iclustering = False # DTW-AROW (default)
            missing_method_partial_dba = False # DTW-AROW (default)
        # dtw_arow_fast = True
        dba_init = dtw_cai_params['dba_init']
        random_state = dtw_cai_params['random_state']
        
        n, m = self._dataset.shape[0:2] # number of time series and number of time samples 
        # TODO: This does not work for time series of different lengths !
        
        if maxiter_dba >= 1: # compute DBA
            dba_func = lambda X_, dba_averages_initial: np.array(dtw_barycenter.dba_loop(X_, c=dba_averages_initial, keep_averages=False, max_it=maxiter_dba, thr=dtw_cai_params['dba_thr'], 
                                                                                        **dtw_params #window=window, psi=psi, missing_normalization=missing_normalization
                                                                                        )
                                                                )
        elif maxiter_dba == 0: # use the medoid (with linear interpolation for missing values) instead of DBA
            def dba_func(X_, dba_averages_initial):
                D_ = dtw_arow_distance_matrix(self.dataset, dtw_params, n_jobs=n_jobs, progress_bar=progress_bar)
                mn = X_[np.argmin(D_.mean(axis=0))] # medoid
                mn_imputed = mus.interpolate_missing(mn) # interpolate because of possible missing values
                return mn_imputed
        
        elif maxiter_dba == -1: # use the mean (with linear interpolation for missing values if necessary) instead of DBA
            def dba_func(X_, dba_averages_initial):
                mn = np.nanmean(self.dataset, axis=0) # mean
                return mus.interpolate_missing(mn) # interpolate because of possible missing values
        
        print('Computing DTW-AROW distances...')
        self.D_dtw_arow = dtw_arow_distance_matrix(self._dataset, dtw_params, n_jobs, progress_bar)
        
        print('Executing the clustering in DTW-CAI...')
        select_columns = lambda O, o: O[np.arange(len(O)), o] # for each row r of O, select o[i]th column

        def find_worst_instance_index(labels_, means_, distances_to_means_): # find the index of the "worst" instance, i.e. the farthest one to the cluster mean in the largest cluster
            c_largest = mode(labels_)[0] # largest cluster
            inds_c_largest = np.where(labels_ == c_largest)[0]
            distances_to_mean_c_largest = select_columns(distances_to_means_[inds_c_largest], labels_[inds_c_largest]) # distances to the mean of the associated cluster (for the largest cluster)
            return inds_c_largest[np.argmax(distances_to_mean_c_largest)]
        
        # determine the number of clusters if necessary:
        elbow_used = False
        if k_iclustering in ['elbow', 'elbow_corrected']: # then use the CORRECTED elbow method (corrected from the commit fcce9ddb3ed954f58a01cd90d265af3326a34a27)
            elbow_used = True
            
            print("Running the elbow method ...")
            try: 
                from yellowbrick.cluster import KElbowVisualizer # breaks matplotlob defaults !!!
                from dtw_missing.tools.clustering_utils import distortion_metric_precomputed
                from yellowbrick.cluster.elbow import KELBOW_SCOREMAP, distortion_score
            except ImportError:
                logger.debug('The yellowbrick library is not available, cannot use the elbow method.')

            KELBOW_SCOREMAP['distortion'] = lambda distance_matrix, labels, *args, **kwargs: distortion_metric_precomputed(distance_matrix, labels)
            model_clustering = KMedoids(metric='precomputed', random_state=555,
                                        init='k-medoids++', # prevents empty clusters
                                        )
            
            visualizer = KElbowVisualizer(model_clustering, 
                                          k=dtw_cai_params['no_clusters_range_for_elbow'], 
                                          timings=False, metric='distortion')
            visualizer.fit(self.D_dtw_arow)
            k_iclustering = visualizer.elbow_value_
            if k_iclustering is None: # no elbow found
                dif = np.diff(visualizer.k_scores_)
                k_iclustering = 1 + visualizer.k_values_[np.nanargmin(dif)] # the location that follows the largest slope
            KELBOW_SCOREMAP['distortion'] = distortion_score
            if show_elbow:
                visualizer.show()
                print('optimal number of clusters determined using elbow method:', k_iclustering)
            
            import matplotlib as mpl
            mpl.rcParams.update(mpl.rcParamsDefault) # restore matplotlib defaults because KElbowVisualizer changes them when imported !!!
        
        # initialize:
        if dba_init is None: # random initialization using the provided random state
            print('DBA initialized randomly')
            rng = np.random.RandomState(self.missing_method_params['random_state']) # fixed seed
            dba_averages_all_iterations = [self.dataset[rng.randint(n, size=k_iclustering)]] # random initialization
        elif dba_init == 'medoid':
            print('DBA initialized by the medoid')
            D_ = self.D_dtw_arow
            kmd = KMedoids(n_clusters = k_iclustering, 
                           metric='precomputed', random_state=555,
                           init='k-medoids++', # prevents empty clusters
                          ).fit(D_)
            dba_averages_all_iterations = [[]]
            for c in range(k_iclustering): # cluster
                inds = np.where(kmd.labels_==c)[0]
                D_c = D_[:, inds]
                dba_averages_all_iterations[0].append(self.dataset[inds[np.argmin(D_c.mean(axis=0))]]) # medoid
        
        labels = [] # labels in each iteration
        Xs_imp = [] # imputed X in each iteration
        distances_to_means = [] # distances of instances to cluster means in each iteration
        inertias = [] # cluster inertias for each iteration
        i = 0
        X_ = copy.deepcopy(self.dataset)
        while True:
            print(f'\nIteration{i}:')
            # Assign labels:
            # labels.append([np.apply_along_axis(lambda o: distance_func(o, X_missing[ii]), 1, means[-1]).argmin() for ii in range(n)]) # slow, non-parallel
            # labels.append([np.apply_along_axis(lambda o: Experiment_DTW_AROW.calculate_distance(o, X_[ii] if iterative_imputation else self.dataset[ii], **self.dtw_params), 1, dba_averages_all_iterations[-1]).argmin() for ii in range(n)]) # slow, non-parallel
            distances_to_means.append(cdist_generic(lambda o, oo: dtw_arow_distance(o, oo, dtw_params), #distance_func, 
                                                    dataset1 = X_ if iterative_imputation else self.dataset, 
                                                    dataset2 = dba_averages_all_iterations[-1], 
                                                    n_jobs=n_jobs, 
                                                    progress_bar=progress_bar,
                                                    # compute_diagonal=False, 
                                                    # parallel_method='processes'
                                                   )
                                     )
            
            labels.append(np.argmin(distances_to_means[-1], axis=1))
            
            inertias.append(np.sum(np.min(distances_to_means[-1], axis=1)))
            
            if iterative_imputation or save_imputed_all_iterations: # If iterative_imputation, then (distances computed based on) imputed data will be used in the next iteration. If save_imputed_all_iterations, then imputed data will be saved (but will not be used if iterative_imputation is False).
                # Impute missing parts using DTW-ID:
                X_ = copy.deepcopy(self.dataset) # reset X_ so that the missing parts can be imputed again with the updated means
                
                dtw_params = copy.deepcopy(dtw_params)
                X_c = Parallel(n_jobs=1)(delayed(dtw_barycenter_imputation.impute_with_dba_multiple)(
                                                                                                  X_[labels[-1] == c], dba_averages_all_iterations[-1][c],
                                                                                                  n_jobs=n_jobs, progress_bar=False, 
                                                                                                  **dtw_params, # window=window, psi=psi, 
                                                                                                 )
                                              for c in tqdm(range(k_iclustering), desc='imputing', disable=not progress_bar)
                                             )
                for c in range(k_iclustering):
                    if np.any(labels[-1] == c): # otherwise no instances associated with cluster c, no need to impute anything
                        X_[labels[-1] == c] = X_c[c]
                
                Xs_imp.append(X_)
            
            can_stop = True # otherwise the algorithm cannot stop
            
            # Handle completely missing cluster means:
            for c in range(k_iclustering):
                if np.isnan(dba_averages_all_iterations[-1][c]).all(): # all time samples are missing
                    print(f'cluster {c} (counting from 0) has a completely missing mean! Assigning another instance to it...')
                    ind_assign = find_worst_instance_index(labels[-1], dba_averages_all_iterations[-1], distances_to_means[-1])
                    # means[-1][c] = X_missing[ind_assign]
                    labels[-1][ind_assign] = c # assign a new instance to cluster c
                    # distances_to_means[-1][ind_assign] = cdist_generic(distance_func, 
                    #                                                    dataset1 = np.atleast_2d(X_missing[ind_assign]), 
                    #                                                    dataset2 = means[-1], 
                    #                                                    n_jobs=-1, 
                    #                                                    progress_bar=True)
                    can_stop = False # cannot stop because the updated labels[-1] is not compatible with means[-1] and distances_to_means[-1]
            
            # Handle empty clusters:
            for c in range(k_iclustering):
                if not any(labels[-1] == c): # empty cluster
                    print(f'cluster {c} (counting from 0) is empty! Assigning another instance to it...')
                    ind_assign = find_worst_instance_index(labels[-1], dba_averages_all_iterations[-1], distances_to_means[-1])
                    labels[-1][ind_assign] = c # assign an instance to cluster c
                    can_stop = False # cannot stop because the updated labels[-1] is not compatible with means[-1] and distances_to_means[-1]
            
            # Stopping condition:
            if can_stop and ( (maxiter_kmeans >= 1 and i >= maxiter_kmeans) or (len(labels) > 1 and np.all(labels[-1] == labels[-2])) ): # can stop and (maxiter_kmeans not specified or reached, or no update in the labels). Note: can stop before if 100th iteration is reached
                #means.pop()
                break
            
            i+=1

            # Update the means:
            # parallel:
            dba_averages_all_iterations.append(np.vstack(Parallel(n_jobs = 1 if maxiter_dba==0 else n_jobs)(delayed(dba_func)(
                                                                                                                            X_[labels[-1]==c] if iterative_imputation else self.dataset[labels[-1]==c], 
                                                                                                                            dba_averages_all_iterations[-1][c]
                                                                                                                           )
                                                         for c in tqdm(range(k_iclustering), desc='updating the means', disable=not progress_bar)))
                                              )
            # # non-parallel way:
            # means_ = np.full((k_iclustering, m), np.nan)
            # for c in tqdm(range(k_iclustering)):
            #     means_[c] = np.array(dtw_barycenter.dba_loop(X_missing[labels[-1]==c], c=means[-1][c], keep_averages=False, max_it=10, thr=0, use_c=False,
            #                                                  use_dtw_missing=True, 
            #                                                  missing_method=missing_method_dba, window=window, psi=psi, missing_normalization=missing_normalization))
            # means.append(means_)

        labels_est = labels[-1] # cluster labels (at the last iteration of k-means)
        
        if not iterative_imputation:
            X_imp = np.full_like(self.dataset, np.nan)
            
            X_imp_c = Parallel(n_jobs=1)(delayed(dtw_barycenter_imputation.impute_with_dba_multiple)(
                                                                                                  self.dataset[labels_est == c], dba_averages_all_iterations[-1][c],
                                                                                                  n_jobs=n_jobs, progress_bar=False, 
                                                                                                  **dtw_params,  # window=window, psi=psi, 
                                                                                                 )
                                              for c in tqdm(range(k_iclustering), desc='imputing', disable=not progress_bar)
                                             )
            for c in range(k_iclustering):
                if np.any(labels_est == c): # otherwise no instances associated with cluster c, no need to impute anything
                    X_imp[labels_est == c] = X_imp_c[c]
        
        # Save the results:
        self.experiment_results_additional = {}
        self.dataset_imputed = Xs_imp[-1] if iterative_imputation else X_imp
        # self.experiment_results_additional['dba_averages_initial'] = dba_averages_initial
        self.experiment_results_additional['dba_averages'] = dba_averages_all_iterations[-1]
        self.experiment_results_additional['dba_averages_all_iterations'] = dba_averages_all_iterations
        # self.experiment_results_additional['distances_to_means'] = distances_to_means # all iterations
        clustering_results = {'k_elbow': k_iclustering, # saved even when the elbow method is not used (in which case this is the same as self.missing_method_params['no_clusters'])
                              'labels_est': labels_est, 
                              # 'ARI_clustering': sklm.adjusted_rand_score(self.y, labels_est)
                             }
        if elbow_used: # then save additional results:
            clustering_results['elbow_k_values'] = visualizer.k_values_
            clustering_results['elbow_k_scores'] = visualizer.k_scores_
        # if self.y is not None:
        #     clustering_results['ARI_clustering'] = sklm.adjusted_rand_score(self.y, labels_est)
        # else:
        #     clustering_results['ARI_clustering'] = None
        if save_additional_clustering_results:
            clustering_results['labels_est_all_iterations'] = labels
            # if self.y is not None:
            #     clustering_results['ARI_clustering_all_iterations'] = [sklm.adjusted_rand_score(self.y, o) for o in labels]
            # else:
            #     clustering_results['ARI_clustering_all_iterations'] = None
            clustering_results['inertia_all_iterations'] = np.array(inertias)
        self.experiment_results_additional['clustering_results'] = clustering_results
        
        self.experiment_results_additional['dataset_imputed'] = self.dataset_imputed # added later, some saved results may not have this
        if save_imputed_all_iterations:
            self.experiment_results_additional['dataset_imputed_all_iterations'] = Xs_imp # corrected later, some saved results may contain incorrect data X_

        print('\nCalculating DRW-AROW distances based on the imputed data...')
        
        self.D = dtw_arow_distance_matrix(self.dataset_imputed, 
                                          dtw_params=dtw_params, 
                                          n_jobs=n_jobs, progress_bar=progress_bar)
        
        print('DTW-CAI completed.')


#TODO: consider adding local baselines: linear interpolation and removal
#TODO: consider adding global existing techniques: k-NN imputation (CRLI not feasible)
#TODO: consider adding a method for introducing missing values