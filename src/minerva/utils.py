"""Metric calculation, objective functions, utilities for optimiastion."""

from typing import Sequence, Optional, Dict, List, Tuple

import numpy as np
import torch

from torch.quasirandom import SobolEngine
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective import is_non_dominated


def compute_hypervolume(ref_point: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute the hypervolume of a set of points with respect to a provided reference point.

    Args:
        ref_point (torch.Tensor): Reference point for hypervolume calculation, typically the nadir point.
        Y (torch.Tensor): A tensor containing multiple points (tensors) over which to compute the hypervolume.

    Returns:
        float: The computed hypervolume.
    """
    bd = NondominatedPartitioning(ref_point=ref_point, Y=Y)
    hypervolume = bd.compute_hypervolume().item()

    return hypervolume


def get_pareto_points(Y: torch.Tensor) -> torch.Tensor:
    """
    Get the Pareto points from a set of points by identifying non-dominated points.

    Args:
        Y (torch.Tensor): A tensor containing multiple points, where each point is represented as a row in the tensor.
                          Each row should represent a point in multi-dimensional space.

    Returns:
        torch.Tensor: A tensor of the Pareto points.
    """

    pareto_mask = is_non_dominated(Y)
    pareto_points = Y[pareto_mask]

    return pareto_points


def compute_log_hypervolume_regret(
    achieved_hypervolume: float, optimal_hypervolume: float
) -> float:
    """
    Computes the logarithm of the hypervolume regret, defined as the difference
    between the optimal and achieved hypervolumes.

    Args:
        achieved_hypervolume (float): The hypervolume achieved by the optimization process.
        optimal_hypervolume (float): The maximum possible or reference hypervolume.

    Returns:
        float: The natural logarithm of the hypervolume regret if positive; otherwise, 0.
    """

    hypervolume_regret = optimal_hypervolume - achieved_hypervolume
    return np.log(hypervolume_regret) if hypervolume_regret > 0 else 0


def calculate_igd(pf_true: np.ndarray, pf_approx: np.ndarray) -> float:
    """
    Calculate the Inverted Generational Distance (IGD) between a true Pareto front and an approximated Pareto front.
    IGD measures the average Euclidean distance from each point in the true Pareto front to the nearest point in the approximated front.

    Args:
        pf_true (array_like): A 2D array where each row represents a point in the true Pareto front.
        pf_approx (array_like): A 2D array where each row represents a point in the approximated Pareto front.

    Returns:
        float: The calculated IGD.
    """

    # Calculate the Euclidean distance from each pareto point to each point in the approximated front, returning the minimum.
    distances = [
        np.min(np.linalg.norm(pf_true - approx_point)) for approx_point in pf_approx
    ]
    igd = np.mean(distances)

    return igd


def igd_plus_distance_maximization(
    true_point: np.ndarray, approx_point: np.ndarray
) -> float:
    """
    Calculate the IGD+ distance for maximization problems from a single reference point to a solution point.

    Args:
    - true_point (array_like): The reference point z = (z1, z2, ..., zm) taken from the true pareto front.
    - approx_point (array_like): The solution point a = (a1, a2, ..., am) taken from the approximate/solution pareto front.

    Returns:
    - float: The IGD+ distance.
    """
    # For maximization: use max(zi - ai, 0), IGD+ measures shortfall euclidean distances rather than actual distance to point, thus accounting for dominance
    distances = [max(z - a, 0) for z, a in zip(true_point, approx_point)]

    # Calculate the Euclidean distance
    return np.sqrt(sum(d**2 for d in distances))


def calculate_igd_plus(pf_true: np.ndarray, pf_approx: np.ndarray) -> float:
    """
    Calculate the Inverted Generational Distance Plus (IGD+) for maximization problems between a true Pareto front and an approximated Pareto front.
    IGD+ in addition to IGD contextualises dominated solutions inside the solution pareto front.

    Args:
    - pf_true (array_like): A 2D array where each row represents a point in the true Pareto front.
    - pf_approx (array_like): A 2D array where each row represents a point in the approximated Pareto front.

    Returns:
    - float: The calculated IGD+.
    """

    distances = [
        min(
            igd_plus_distance_maximization(true_point, approx_point)
            for approx_point in pf_approx
        )
        for true_point in pf_true
    ]
    return np.mean(distances)


def count_identified_pareto_points(
    pareto_ground_truth: torch.Tensor,
    search_strategy_points: torch.Tensor,
    tolerance: float = 1e-6,
) -> int:
    """
    Count the number of ground truth Pareto points identified by a search strategy.

    Args:
        pareto_ground_truth (torch.Tensor): Tensor of ground truth Pareto points.
        search_strategy_points (torch.Tensor): Tensor of points sampled by the search strategy.
        tolerance (float): Numeric tolerance for considering two points as identical.

    Returns:
        int: Number of identified ground truth Pareto points.
    """

    identified_count = 0
    for ground_truth_point in pareto_ground_truth:
        # Check if this ground truth point is in the search strategy points
        differences = search_strategy_points - ground_truth_point
        is_close = torch.all(torch.abs(differences) <= tolerance, dim=1)

        if torch.any(is_close):
            identified_count += 1

    return identified_count


def draw_sobol_samples(n_samples: int, feature_matrix: torch.Tensor) -> torch.Tensor:
    """
    Draw Sobol samples from a pre-defined, normalized (0,1) feature space and finds the closest points in the feature matrix.

    The feature matrix is pre-normalized to ensure same scale as Sobol.

    Args:
        n_samples (int): Number of Sobol samples to draw.
        feature_matrix (torch.Tensor): A tensor representing the feature space, normalized to (0,1), where each row
                                       is a feature vector.

    Returns:
        torch.Tensor: The closest points in the feature matrix corresponding to each Sobol sample.

    Note:
        This function's reproducibility is influenced by the global random seed setting and scrambling of Sobol sequences.
    """

    num_features = feature_matrix.size(1)
    sobol_sampler = SobolEngine(dimension=num_features, scramble=True)

    # Generate Sobol samples in unit hypercube
    sobol_samples = sobol_sampler.draw(n_samples).to(feature_matrix)

    # Find closest points in the tensor for each Sobol sample
    selected_indices = torch.cdist(sobol_samples, feature_matrix).argmin(dim=1)
    selected_configurations = feature_matrix[selected_indices]

    return selected_configurations


def tensor_to_tuple(t: torch.Tensor) -> Tuple:
    """
    Convert a tensor to a tuple for use as a dictionary key.
    This is required as dictionary keys are required to be immutable.
    However, tensor values can be updated so should be converted.

    Args:
        t (torch.Tensor): A PyTorch tensor.

    Returns:
        tuple: A tuple representation of the input tensor.
    """
    return tuple(t.tolist())


def create_lookup_dict(X_lookup: torch.Tensor) -> Dict:
    """
    Create a lookup dictionary from an iterable of tensors to their indices.

    Args:
        X_lookup [torch.Tensor]: An iterable tensor where each row is used as lookup values.

    Returns:
        dict: A dictionary mapping tensor tuples to their respective indices.
    """
    return {tensor_to_tuple(x): i for i, x in enumerate(X_lookup)}


def unified_objective_function(
    X: torch.Tensor,
    X_lookup: torch.Tensor,
    Y_lookup: torch.Tensor,
    noise_stds: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Maps each tensor in X to a given index in X_lookup, and returns the value of Y_lookup[index].

    Args:
        X (torch.Tensor): A tensor containing the items for which corresponding Y_lookup values are desired
        X_lookup (torch.Tensor): A tensor where each row represents a unique key for lookup.
                                 Each row in X should have a corresponding row in X_lookup.
        Y_lookup (torch.Tensor): A tensor where each row contains values corresponding to the keys in X_lookup.
                                 The index of each row in Y_lookup should match the index of the corresponding row in X_lookup.

    Returns:
        torch.Tensor: A tensor containing the Y_lookup values corresponding to each tensor in X, in the order they
                      are found in X. If a tensor in X does not have a corresponding value (i.e., it is not found in
                      X_lookup), it is omitted from the result.
    """
    # get indices of X in X_lookup, this is a list
    indices = get_index_from_lookup(X, X_lookup)

    # get the corresponding Y values
    results = torch.stack([Y_lookup[idx] for idx in indices])

    if noise_stds:
        results = add_noise_to_objectives(results, noise_stds)
    return results


def add_noise_to_objectives(
    objectives: torch.Tensor, noise_stds: Sequence[float]
) -> torch.Tensor:
    """
    Adds Gaussian noise to each objective in the objectives tensor.

    Args:
        objectives (torch.Tensor): Tensor of objectives where each column corresponds to a different objective.
        noise_stds (Sequence[float]): Standard deviations for the Gaussian noise to be added to each objective.

    Returns:
        torch.Tensor: Tensor of objectives with added noise.
    """
    noisy_objectives = []
    for i, std in enumerate(noise_stds):
        if std > 0:
            noise = torch.randn_like(objectives[:, i]) * std
            noisy_obj = torch.clamp(
                objectives[:, i] + noise,
                min=0,
                max=100 if i == 0 else torch.max(objectives[:, i]),
            )
        else:
            noisy_obj = objectives[:, i]
        noisy_objectives.append(noisy_obj.unsqueeze(1))
    return torch.cat(noisy_objectives, dim=1)


def get_index_from_lookup(X: torch.Tensor, X_lookup: torch.Tensor) -> List:
    """
    Finds the index (or indices) of elements in `X` within `X_lookup`.

    Args:
        X (torch.Tensor): A tensor containing elements to look up.
        X_lookup (torch.Tensor): A tensor serving as the lookup source.

    Returns:
        List: Indices of where X is found in X_lookup.

    This function facilitates finding the position(s) of tensor(s) in a lookup tensor, useful for indexing or cross-referencing operations.
    """
    # Creates a lookup dictionary to match tensors in X_lookup to their indices
    lookup_dict = create_lookup_dict(X_lookup)

    if X.ndim == 1:
        X = X.unsqueeze(0)

    index_list = []
    for x in X:
        key = tensor_to_tuple(x)
        if key in lookup_dict:
            index_list.append(lookup_dict[key])

    return index_list


def filter_tensor_random(
    search_matrix: torch.Tensor,
    search_labels: torch.Tensor,
    feature_index: int,
    n_unique_features: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filters the search matrix and labels to only include rows where the specified feature
    values are randomly selected from the unique values of that feature.

    Given a matrix of experimental configurations (search_matrix) and their corresponding
    objective function values (search_labels), this function filters the configurations
    based on the specified feature.

    For values of n_unique_features, it randomly selects the specified number of unique
    feature values and includes only the configurations that match these selected feature values.

    Args:
        search_matrix (torch.Tensor): A tensor containing different experimental configurations.
        search_labels (torch.Tensor): A tensor containing the objective function values corresponding to
                                      the configurations in search_matrix.
        feature_index (int): The index of the feature based on which the filtering will be performed.
        n_unique_features (int, optional): The number of unique feature values to filter by. Defaults to 2.

    Returns:
        Tuple: A tuple containing the filtered search matrix and labels.

    Note:
        The function supports filtering for any number of unique features specified by n_unique_features.
        If n_unique_features is larger than the number of unique features available, it defaults to using
        all unique features.
    """
    constrained_feature = search_matrix[:, feature_index]
    unique_values = torch.unique(constrained_feature)

    if n_unique_features < unique_values.shape[0]:
        # Randomly select n_unique_features from the unique values
        selected_indices = torch.randperm(unique_values.shape[0])[:n_unique_features]
        selected_values = unique_values[selected_indices]
        # print(selected_values)

        # Construct a mask for rows with the selected feature values, creating same number of rows as the actual tensor data.
        mask = torch.zeros_like(constrained_feature, dtype=torch.bool)
        for value in selected_values:
            mask |= constrained_feature == value

        # mask tells which tensor rows to keep and discard!
        return search_matrix[mask], search_labels[mask]

    else:
        return search_matrix, search_labels
