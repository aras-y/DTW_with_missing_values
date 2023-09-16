# taken from the gist https://gist.github.com/rtavenar/a4fb580ae235cc61ce8cf07878810567 of Romain Tavenard on 07.09.2021
# 
# modified: a progress bar and joblib parallel processing method selection added


import numpy
from joblib.parallel import Parallel, delayed
from tqdm import tqdm

def cdist_generic(dist_fun, dataset1, dataset2=None, n_jobs=None, verbose=0,
                  compute_diagonal=True, 
                  progress_bar=False, parallel_method='processes',
                  *args, **kwargs):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.

    Parameters
    ----------
    dist_fun : function
        Similarity function to be used. Should be a function such that
        `dist_fun(dataset1[i], dataset2[j])` returns a distance (a float).

    dataset1 : array-like
        A dataset

    dataset2 : array-like (default: None)
        Another dataset. If `None`, self-similarity of `dataset1` is returned.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`_
        for more details.

    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.
    
    progress_bar : bool (default: False)
        show a progress bar
    
    parallel_method : str (default: 'processes')
        'processes', 'threads', ... (see joblib.Parallel)

    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.

    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix
    """ # noqa: E501
    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = numpy.zeros((len(dataset1), len(dataset1)))
        indices = numpy.triu_indices(len(dataset1),
                                     k=0 if compute_diagonal else 1,
                                     m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs,
                                   prefer=parallel_method,
                                   verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset1[j],
                *args, **kwargs
            )
            for i in tqdm(range(len(dataset1)), disable=not progress_bar)
            for j in range(i if compute_diagonal else i + 1,
                           len(dataset1))
        )
        indices = numpy.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        return matrix
    else:
        matrix = Parallel(n_jobs=n_jobs, prefer=parallel_method, verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset2[j],
                *args, **kwargs
            )
            for i in tqdm(range(len(dataset1)), disable=not progress_bar, desc='calculating distances') for j in range(len(dataset2))
        )
        return numpy.array(matrix).reshape((len(dataset1), -1))