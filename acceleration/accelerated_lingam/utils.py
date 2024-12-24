import numbers
import numpy as np

from sklearn.linear_model import LassoLarsIC

__all__ = ['get_kernel_width', 'get_gram_matrix', 'hsic_teststat', \
           'hsic_test_gamma', 'predict_adaptive_lasso', 'find_all_paths']

import numpy as np
from math import ceil, exp, gamma

def gamma_pdf(x, a, b):
    if x < 0:
        return 0
    else:
        return (x**(a-1) * exp(-x/b)) / (b**a * gamma(a))

def gamma_cdf(x, a, b):
    if x < 0:
        return 0
    else:
        n = int(ceil(x/b))
        h = x / (b * n)
        sum = 0
        for i in range(n):
            x_i = i * h
            sum += 0.5 * (gamma_pdf(x_i, a, b) + gamma_pdf(x_i + h, a, b)) * h
        return sum



def linear_regression(X, y=None, fit_intercept=True):
    # Add intercept term if required
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Compute SVD of X
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    # Compute pseudoinverse of X
    S_inv = np.diag(1 / S)
    X_pseudo_inv = VT.T @ S_inv @ U.T

    # Compute coefficients
    coef = X_pseudo_inv @ y

    return coef


def scale(X):
    """Centers and scales the input features."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_centered = X - mean
    X_scaled = X_centered / std
    return X_scaled


def resample(X, y, n_samples=None, replace=True, stratify=None, random_state=None):
    """Resample arrays or sparse matrices.

    Parameters:
    ----------
    X : {ndarray, sparse matrix}
        The data to resample. Must be 2D, with the number of samples
        in the first dimension and the number of features in the second dimension.
    y : {ndarray, int, None}
        The labels, if any. Must be 1D, with the number of labels
        equal to the number of samples in X. If None, the labels
        are not resampled.
    n_samples : int, default=None
        The number of samples to resample. If None, the length of X is
        used. If replace is False, the number of samples must be less
        than or equal to the length of X.
    replace : bool, default=True
        Whether to resample with replacement. If True, samples are drawn
        with replacement, which means that the same sample can be drawn
        multiple times. If False, samples are drawn without replacement,
        which means that each sample can only be drawn once.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this
        as the class labels. This argument is ignored if y is None.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling the data.
        Pass an int for reproducible results across multiple function calls.
        See Glossary.

    Returns:
    ----------
    X_resampled : array or sparse matrix
        The resampled data.
    y_resampled : array or None
        The resampled labels, if any.
    """

    if n_samples is None:
        n_samples = X.shape[0]

    if replace and n_samples > X.shape[0]:
        raise ValueError("Number of samples must be less than or equal to the length of X if replace is True.")

    indices = np.random.choice(X.shape[0], size=n_samples, replace=replace, random_state=random_state)

    X_resampled = X[indices]

    if stratify is not None:
        strata, counts = np.unique(stratify, return_counts=True)
        n_samples_per_stratum = np.floor(n_samples * counts / np.sum(counts))

        indices_per_stratum = []
        for stratum, n_samples in zip(strata, n_samples_per_stratum):
            indices_stratum = np.random.choice(np.where(stratify == stratum)[0], size=int(n_samples), replace=False, random_state=random_state)
            indices_per_stratum.append(indices_stratum)

        indices = np.concatenate(indices_per_stratum)

    if y is not None:
        y_resampled = y[indices]

    return X_resampled, y_resampled


def check_array(X, accept_sparse=False, copy=False, ensure_2d=True, allow_nd=False, dtype=None, order=None, force_all_finite=False):
    """Check the array and cast it if necessary.

    Parameters
    ----------
    X : array-like
        The array to check.

    accept_sparse : bool, default=False
        If True, allows sparse matrices.

    copy : bool, default=False
        If True, the data will be copied, otherwise it may be modified
        in-place.

    ensure_2d : bool, default=True
        If True, ensures 2-dimensionality of the input. Otherwise,
        unchanged.

    allow_nd : bool, default=False
        If True, accepts N-dimensional arrays. Otherwise, checks the
        dimensionality of the input.

    dtype : dtype, default=None
        Data type of the result. If None, the data type of the
        input is preserved.

    order : {'C', 'F', 'A'}, default=None
        Order of the result. If None, the order of the input is preserved.

    force_all_finite : bool, default=False
        If True, forces all values of the input to be finite.
        Otherwise, values of the input may be NaN or inf.

    Returns
    -------
    X_new : array
        A checked and cast copy of the array X.

    """

    if hasattr(X, "toarray"):
        # if a sparse matrix, convert it to a dense array
        X = X.toarray(copy=copy)
    else:
        # ensure X is a numpy array
        X = np.asarray(X, order=order)

    if ensure_2d and X.ndim < 2:
        X = X.reshape(1, X.shape[0])

    if force_all_finite:
        X_finite = np.isfinite(X)
        if not X_finite.all():
            raise ValueError("Some of the values in X are not finite.")

    if dtype is not None:
        X = X.astype(dtype, copy=False, order=order)

    return X



def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    # X = X.astype(np.float64)
    coef = linear_regression(X[:, predictors], X[:, target])[1:]
    weight = np.power(np.abs(coef), gamma)

    reg = LassoLarsIC(criterion='bic', max_iter=100)
    reg.fit(X[:, predictors] * weight, X[:, target])

    return reg.coef_ * weight


def find_all_paths(dag, from_index, to_index, min_causal_effect=0.0):
    """Find all paths from point to point in DAG.

    Parameters
    ----------
    dag : array-like, shape (n_features, n_features)
        The adjacency matrix to fine all paths, where n_features is the number of features.
    from_index : int
        Index of the variable at the start of the path.
    to_index : int
        Index of the variable at the end of the path.
    min_causal_effect : float, optional (default=0.0)
        Threshold for detecting causal direction.
        Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

    Returns
    -------
    paths : array-like, shape (n_paths)
        List of found path, where n_paths is the number of paths.
    effects : array-like, shape (n_paths)
        List of causal effect, where n_paths is the number of paths.
    """
    # Extract all edges
    edges = np.array(np.where(np.abs(np.nan_to_num(dag)) > min_causal_effect)).T

    # Aggregate edges by start point
    to_indices = []
    for i in range(dag.shape[0]):
        adj_list = edges[edges[:, 1] == i][:, 0].tolist()
        if len(adj_list) != 0:
            to_indices.append(adj_list)
        else:
            to_indices.append([])

    # DFS
    paths = []
    stack = [from_index]
    stack_to_indice = [to_indices[from_index]]
    while stack:
        if len(stack) > dag.shape[0]:
            raise ValueError(
                "Unable to find the path because a cyclic graph has been specified.")

        cur_index = stack[-1]
        to_indice = stack_to_indice[-1]

        if cur_index == to_index:
            paths.append(stack.copy())
            stack.pop()
            stack_to_indice.pop()
        else:
            if len(to_indice) > 0:
                next_index = to_indice.pop(0)
                stack.append(next_index)
                stack_to_indice.append(to_indices[next_index].copy())
            else:
                stack.pop()
                stack_to_indice.pop()

    # Calculate the causal effect for each path
    effects = []
    for p in paths:
        coefs = [dag[p[i + 1], p[i]] for i in range(len(p) - 1)]
        effects.append(np.cumprod(coefs)[-1])

    return paths, effects


class BootstrapMixin():
    """Mixin class for all LiNGAM algorithms that implement the method of bootstrapping."""

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        for i in range(n_sampling):
            self.fit(resample(X))
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1:]:
                    total_effects[i, to, from_] = self.estimate_total_effect(
                        X, from_, to)

        return BootstrapResult(adjacency_matrices, total_effects)


class BootstrapResult(object):
    """The result of bootstrapping."""

    def __init__(self, adjacency_matrices, total_effects):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        """
        self._adjacency_matrices = adjacency_matrices
        self._total_effects = total_effects

    @property
    def adjacency_matrices_(self):
        """The adjacency matrix list by bootstrapping.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (n_sampling)
            The adjacency matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._adjacency_matrices

    @property
    def total_effects_(self):
        """The total effect list by bootstrapping.

        Returns
        -------
        total_effects_ : array-like, shape (n_sampling)
            The total effect list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._total_effects

    def get_causal_direction_counts(self, n_directions=None, min_causal_effect=None, split_by_causal_effect_sign=False):
        """Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        causal_direction_counts : dict
            List of causal directions sorted by count in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'count': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if isinstance(n_directions, (numbers.Integral, np.integer)):
            if not 0 < n_directions:
                raise ValueError(
                    'n_directions must be an integer greater than 0')
        elif n_directions is None:
            pass
        else:
            raise ValueError('n_directions must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Count causal directions
        directions = []
        for am in np.nan_to_num(self._adjacency_matrices):
            direction = np.array(np.where(np.abs(am) > min_causal_effect))
            if split_by_causal_effect_sign:
                signs = np.array([np.sign(am[i][j])
                                  for i, j in direction.T]).astype('int64').T
                direction = np.vstack([direction, signs])
            directions.append(direction.T)
        directions = np.concatenate(directions)

        if len(directions) == 0:
            cdc = {'from': [], 'to': [], 'count': []}
            if split_by_causal_effect_sign:
                cdc['sign'] = []
            return cdc

        directions, counts = np.unique(directions, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_directions] if n_directions is not None else sort_order
        counts = counts[sort_order]
        directions = directions[sort_order]

        cdc = {
            'from': directions[:, 1].tolist(),
            'to': directions[:, 0].tolist(),
            'count': counts.tolist()
        }
        if split_by_causal_effect_sign:
            cdc['sign'] = directions[:, 2].tolist()

        return cdc

    def get_directed_acyclic_graph_counts(self, n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=False):
        """Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        directed_acyclic_graph_counts : dict
            List of directed acyclic graphs sorted by count in descending order.
            The dictionary has the following format::

            {'dag': [n_dags], 'count': [n_dags]}.

            where ``n_dags`` is the number of directed acyclic graphs.
        """
        # Check parameters
        if isinstance(n_dags, (numbers.Integral, np.integer)):
            if not 0 < n_dags:
                raise ValueError('n_dags must be an integer greater than 0')
        elif n_dags is None:
            pass
        else:
            raise ValueError('n_dags must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Count directed acyclic graphs
        dags = []
        for am in np.nan_to_num(self._adjacency_matrices):
            dag = np.abs(am) > min_causal_effect
            if split_by_causal_effect_sign:
                direction = np.array(np.where(dag))
                signs = np.zeros_like(dag).astype('int64')
                for i, j in direction.T:
                    signs[i][j] = np.sign(am[i][j]).astype('int64')
                dag = signs
            dags.append(dag)

        dags, counts = np.unique(dags, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
        counts = counts[sort_order]
        dags = dags[sort_order]

        if split_by_causal_effect_sign:
            dags = [{
                'from': np.where(dag)[1].tolist(),
                'to': np.where(dag)[0].tolist(),
                'sign': [dag[i][j] for i, j in np.array(np.where(dag)).T]} for dag in dags]
        else:
            dags = [{
                'from': np.where(dag)[1].tolist(),
                'to': np.where(dag)[0].tolist()} for dag in dags]

        return {
            'dag': dags,
            'count': counts.tolist()
        }

    def get_probabilities(self, min_causal_effect=None):
        """Get bootstrap probability.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        probabilities : array-like
            List of bootstrap probability matrix.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        adjacency_matrices = np.nan_to_num(self._adjacency_matrices)
        shape = adjacency_matrices[0].shape
        bp = np.zeros(shape)
        for B in adjacency_matrices:
            bp += np.where(np.abs(B) > min_causal_effect, 1, 0)
        bp = bp / len(adjacency_matrices)

        if int(shape[1] / shape[0]) == 1:
            return bp
        else:
            return np.hsplit(bp, int(shape[1] / shape[0]))

    def get_total_causal_effects(self, min_causal_effect=None):
        """Get total effects list.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        total_causal_effects : dict
            List of bootstrap total causal effect sorted by probability in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'effect': [n_directions], 'probability': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError(
                    'min_causal_effect must be an value greater than 0.')

        # Calculate probability
        probs = np.sum(np.where(np.abs(self._total_effects) >
                                min_causal_effect, 1, 0), axis=0, keepdims=True)[0]
        probs = probs / len(self._total_effects)

        # Causal directions
        dirs = np.array(np.where(np.abs(probs) > 0))
        probs = probs[dirs[0], dirs[1]]

        # Calculate median effect without zero
        effects = np.zeros(dirs.shape[1])
        for i, (to, from_) in enumerate(dirs.T):
            idx = np.where(np.abs(self._total_effects[:, to, from_]) > 0)
            effects[i] = np.median(self._total_effects[:, to, from_][idx])

        # Sort by probability
        order = np.argsort(-probs)
        dirs = dirs.T[order]
        effects = effects[order]
        probs = probs[order]

        ce = {
            'from': dirs[:, 1].tolist(),
            'to': dirs[:, 0].tolist(),
            'effect': effects.tolist(),
            'probability': probs.tolist()
        }

        return ce

    def get_paths(self, from_index, to_index, min_causal_effect=0.0):
        """Get all paths from the start variable to the end variable and their bootstrap probabilities.

        Parameters
        ----------
        from_index : int
            Index of the variable at the start of the path.
        to_index : int
            Index of the variable at the end of the path.
        min_causal_effect : float, optional (default=0.0)
            Threshold for detecting causal direction.
            Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        paths : dict
            List of path and bootstrap probability.
            The dictionary has the following format::

            {'path': [n_paths], 'effect': [n_paths], 'probability': [n_paths]}

            where ``n_paths`` is the number of paths.
        """
        # Find all paths from from_index to to_index
        paths_list = []
        effects_list = []
        for am in self._adjacency_matrices:
            paths, effects = find_all_paths(am, from_index, to_index)
            # Convert path to string to make them easier to handle.
            paths_list.extend(['_'.join(map(str, p)) for p in paths])
            effects_list.extend(effects)

        paths_list = np.array(paths_list)
        effects_list = np.array(effects_list)

        # Count paths
        paths_str, counts = np.unique(paths_list, axis=0, return_counts=True)

        # Sort by count
        order = np.argsort(-counts)
        probs = counts[order] / len(self._adjacency_matrices)
        paths_str = paths_str[order]

        # Calculate median of causal effect for each path
        effects = [np.median(effects_list[np.where(paths_list == p)])
                   for p in paths_str]

        result = {
            'path': [[int(i) for i in p.split('_')] for p in paths_str],
            'effect': effects,
            'probability': probs.tolist(),
        }
        return result


def get_kernel_width(X):
    """Calculate the bandwidth to median distance between points.
    Use at most 100 points (since median is only a heuristic,
    and 100 points is sufficient for a robust estimate).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    Returns
    -------
    float
        The bandwidth parameter.
    """
    n_samples = X.shape[0]
    if n_samples > 100:
        X_med = X[:100, :]
        n_samples = 100
    else:
        X_med = X

    G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
    dists = G + G.T - 2 * np.dot(X_med, X_med.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n_samples ** 2, 1)

    return np.sqrt(0.5 * np.median(dists[dists > 0]))

def _rbf_dot(X, Y, width):
    """Compute the inner product of radial basis functions."""
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    G = np.sum(X * X, 1).reshape(n_samples_X, 1)
    H = np.sum(Y * Y, 1).reshape(n_samples_Y, 1)
    Q = np.tile(G, (1, n_samples_Y))
    R = np.tile(H.T, (n_samples_X, 1))
    H = Q + R - 2 * np.dot(X, Y.T)

    return np.exp(-H / 2 / (width ** 2))

def _rbf_dot_XX(X, width):
    """rbf dot, in special case with X dot X"""
    G = np.sum(X * X, axis=1)
    H = G[None, :] + G[:, None] - 2 * np.dot(X, X.T)
    return np.exp(-H / 2 / (width ** 2))

def get_gram_matrix(X, width):
    """Get the centered gram matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    width : float
        The bandwidth parameter.

    Returns
    -------
    K, Kc : array
        the centered gram matrices.
    """
    n = X.shape[0]

    K = _rbf_dot_XX(X, width)
    K_colsums = K.sum(axis=0)
    K_rowsums = K.sum(axis=1)
    K_allsum = K_rowsums.sum()
    Kc = K - (K_colsums[None, :] + K_rowsums[:, None]) / n + (K_allsum / n ** 2)
    # equivalent to H @ K @ H, where H = np.eye(n) - 1 / n * np.ones((n, n)).
    return K, Kc


def hsic_teststat(Kc, Lc, n):
    """get the HSIC statistic.

    Parameters
    ----------
    K, Kc : array
        the centered gram matrices.

    n : float
        the number of samples.

    Returns
    -------
    float
        the HSIC statistic.
    """
    # test statistic m*HSICb under H1
    return 1 / n * np.sum(Kc.T * Lc)


def hsic_test_gamma(X, Y, bw_method='mdbs'):
    """get the HSIC statistic.

    Parameters
    ----------
    X, Y : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.

    bw_method : str, optional (default=``mdbs``)
        The method used to calculate the bandwidth of the HSIC.

        * ``mdbs`` : Median distance between samples.
        * ``scott`` : Scott's Rule of Thumb.
        * ``silverman`` : Silverman's Rule of Thumb.

    Returns
    -------
    test_stat : float
        the HSIC statistic.

    p : float
        the HSIC p-value.
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    # Get kernel width to median distance between points
    width_x = get_kernel_width(X)
    width_y = get_kernel_width(Y)

    # these are slightly biased estimates of centered gram matrices
    K, Kc = get_gram_matrix(X, width_x)
    L, Lc = get_gram_matrix(Y, width_y)

    # test statistic m*HSICb under H1
    n = X.shape[0]
    test_stat = hsic_teststat(Kc, Lc, n)

    var = (1 / 6 * Kc * Lc) ** 2
    # second subtracted term is bias correction
    var = 1 / n / (n - 1) * (np.sum(var) - np.trace(var))
    # variance under H0
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    K[np.diag_indices(n)] = 0
    L[np.diag_indices(n)] = 0
    mu_X = 1 / n / (n - 1) * K.sum()
    mu_Y = 1 / n / (n - 1) * L.sum()
    # mean under H0
    mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

    alpha = mean ** 2 / var
    # threshold for hsicArr*m
    beta = var * n / mean
    p = 1 - gamma_cdf(test_stat, alpha, beta)

    return test_stat, p
