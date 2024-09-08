"""Thompson sampling and nested acquisition module."""

import gc
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from gpytorch.likelihoods import Likelihood
from botorch.models import GenericDeterministicModel
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import _split_batch_eval_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler, MCSampler

from minerva.acqfunc import optimize_acqf_and_get_suggestion

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def _batch_sample_posterior_botorch(
    model: ModelListGP, evaluation_matrix: torch.Tensor, max_batch_sample: int = 10000
) -> torch.Tensor:
    """
    Batch sampling from GP posterior when evaluation matrices are large and 
    computational constraints prevent sampling all points at once.

    Uses BoTorch functionality.

    Args:
        model (ModelListGP): The trained GP model object from BoTorch.
        evaluation_matrix (torch.Tensor): A tensor representing the matrix of points to evaluate the model's posterior.
        max_batch_sample (int): The maximum number of samples to process in each batch. The default is 10000.

    Returns:
        torch.Tensor: A tensor containing the sampled values from the model's posterior over the evaluation matrix.
    """

    # splits the evaluation matrix into number of batches with remainder.
    num_batches = len(evaluation_matrix) // max_batch_sample + (
        len(evaluation_matrix) % max_batch_sample > 0
    )
    Y_samples = []

    with torch.no_grad():
        for b in range(num_batches):
            batch_start = b * max_batch_sample
            batch_end = min((b + 1) * max_batch_sample, len(evaluation_matrix))

            batch_matrix = evaluation_matrix[batch_start:batch_end]

            # botorch.posteriors.posterior.sample samples with torch.no_grad
            samples = model.posterior(batch_matrix).sample(torch.Size([1])).squeeze(0)
            Y_samples.append(samples)

            torch.cuda.empty_cache()

        return torch.cat(Y_samples, dim=0)


def _batch_sample_posterior_gpytorch(
    model: ModelListGP,
    likelihood: List[Likelihood],
    evaluation_matrix: torch.Tensor,
    max_batch_sample: int = 10000,
) -> torch.Tensor:
    """
    Batch sampling from GP posterior when evaluation matrices are large and 
    computational constraints prevent sampling all points at once.

    Uses GPyTorch functionality.

    Args:
        model (ModelListGP): The trained multi-output GP model object from BoTorch, with models for each output.
        likelihood (List[Likelihood]): The likelihoods corresponding to each output of the multi-output GP model.
        evaluation_matrix (torch.Tensor): A tensor representing the matrix of points to evaluate the model's posterior.
        max_batch_sample (int): The maximum number of samples to process in each batch. The default is 10000.

    Returns:
        torch.Tensor: A tensor containing the sampled values from each model's posterior over the evaluation_matrix.
    """
    # Calculate the number of batches needed to process all points
    num_batches = len(evaluation_matrix) // max_batch_sample + (
        len(evaluation_matrix) % max_batch_sample > 0
    )
    Y_samples = []

    with torch.no_grad():
        # Iterate over each output model
        for i, single_model in enumerate(model.models):
            model_samples = []
            # Process each batch
            for b in range(num_batches):
                batch_start = b * max_batch_sample
                batch_end = min((b + 1) * max_batch_sample, len(evaluation_matrix))
                batch_matrix = evaluation_matrix[batch_start:batch_end]

                # Sample from the posterior of the current model
                posterior = single_model(batch_matrix)
                observed_pred = likelihood[i](posterior)
                samples = observed_pred.sample(torch.Size([1])).squeeze(0)
                model_samples.append(samples)

                torch.cuda.empty_cache()

            # Combine samples from all batches for the current model
            combined_samples = torch.cat(model_samples, dim=0)
            Y_samples.append(combined_samples)

    return torch.stack(Y_samples, dim=1).squeeze(1)


def _sample_model_posterior(
    model: ModelListGP,
    likelihood: List[Likelihood],
    evaluation_matrix: torch.Tensor,
    max_batch_samples: int = 10000,
    posterior_sampling: str = "TS_botorch",
) -> torch.Tensor:
    """
    Abstracts sampling process from the model's posterior distribution.

    Args:
        model (ModelListGP): The trained GP model object from BoTorch.
        likelihood (List[Likelihood]): The likelihoods corresponding to each output of the multi-output GP model.
        evaluation_matrix (torch.Tensor): A tensor representing the matrix of points to evaluate the model's posterior.
        max_batch_samples (int): The maximum number of samples to process in each batch. The default is 10000.
        posterior_sampling (str): The method to use for sampling from the model's posterior. The default is 'TS_botorch'.

    Returns:
        torch.Tensor: A tensor containing the sampled values from the model's posterior over the evaluation_matrix.
    """

    if posterior_sampling == "TS_botorch":
        Y_samples = _batch_sample_posterior_botorch(
            model, evaluation_matrix, max_batch_sample=max_batch_samples
        )

    elif posterior_sampling == "TS_gpytorch":
        Y_samples = _batch_sample_posterior_gpytorch(
            model, likelihood, evaluation_matrix, max_batch_sample=max_batch_samples
        )

    elif posterior_sampling == "exploitative":
        Y_samples = model.posterior(evaluation_matrix).mean

    return Y_samples


def setup_TS_HVI_and_optimize(
    evaluation_matrix: torch.Tensor,
    train_Y_std: torch.Tensor,
    train_and_pending_Y: torch.Tensor,
    Y_samples: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Set up and optimize TS-HVI for Bayesian Optimization.

    This function draws samples from the posterion, computing a 
    single sample approximation of the hypervolume improvement.
    This is used to select the next point to evaluate.

    Args:
        evaluation_matrix (torch.Tensor): Tensor containing all possible points for evaluating the acquisition.
        train_Y_std (torch.Tensor): Tensor containing the standardized training output data.
        train_and_pending_Y (torch.Tensor): Tensor containing the training and pending output data.
        Y_samples (torch.Tensor): A tensor containing the sampled values from the model's posterior over the evaluation_matrix.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the next point to evaluate and its index.
    """

    ref_point, ref_index = torch.min(train_Y_std, axis=0)

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_and_pending_Y)

    sampled_model = GenericDeterministicModel(
        f=lambda X: Y_samples.unsqueeze(1), num_outputs=Y_samples.shape[-1]
    )

    acqf = qExpectedHypervolumeImprovement(
        model=sampled_model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(num_samples=1),
    )

    with torch.no_grad():
        hvi = acqf(evaluation_matrix.unsqueeze(1)).to(**tkwargs)

    # Select the best candidate based on HVI
    best_index = hvi.argmax()
    candidate = evaluation_matrix[best_index, :].unsqueeze(0)

    return candidate, best_index


def optimize_TS_HVI_and_get_suggestion(
    model: ModelListGP,
    likelihood: List[Likelihood],
    search_matrix: torch.Tensor,
    train_Y_std: torch.Tensor,
    batch_size: int,
    posterior_sampling: str = "TS_botorch",
) -> torch.Tensor:
    """
    Given a trained model and search space, runs one iteration of Bayesian Optimization
    with TS-HVI strategy.

    Args:
        model (Model): A Botorch Gaussian Process model fitted on training data.
        likelihood (List[Likelihood]): The likelihoods corresponding to each output of the multi-output GP model.
        search_matrix (torch.Tensor): A tensor containing all possible points where the acquisition function
                                        should be evaluated to determine the next suggested point.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
        batch_size (int): Number of candidates to generate.
        posterior_sampling (str): The method to use for sampling from the model's posterior. The default is 'TS_botorch'.

    Returns:
        torch.Tensor: Returns a batch of points selected by Bayesian Optimization.
    """

    X_next = torch.empty((0, search_matrix.size(1)), **tkwargs)
    evaluation_matrix = search_matrix.clone()

    train_and_pending_Y = train_Y_std.detach().clone()

    for i in tqdm(range(batch_size)):
        Y_samples = _sample_model_posterior(
            model=model,
            likelihood=likelihood,
            evaluation_matrix=evaluation_matrix,
            max_batch_samples=10000,
            posterior_sampling=posterior_sampling,
        )

        candidate, best_index = setup_TS_HVI_and_optimize(
            evaluation_matrix=evaluation_matrix,
            train_Y_std=train_Y_std,
            train_and_pending_Y=train_and_pending_Y,
            Y_samples=Y_samples,
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        train_and_pending_Y = torch.cat(
            (train_and_pending_Y, Y_samples[best_index, :].unsqueeze(0)), dim=0
        )

        # Remove the selected candidate from the search_matrix
        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

    return X_next


def naive_optimize_TS_HVI_and_get_suggestion(
    model: ModelListGP,
    likelihood: List[Likelihood],
    search_matrix: torch.Tensor,
    train_Y_std: torch.Tensor,
    batch_size: int,
    n_unique_features: int,
    feature_index: int,
    posterior_sampling: str = "TS_botorch",
) -> torch.Tensor:
    """
    Given a trained model and search space, runs one naive constrained iteration of Bayesian Optimization
    with the TS-HVI acquisition function

    Args:
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        likelihood (List[Likelihood]): The likelihoods corresponding to each output of the multi-output GP model.
        search_matrix (torch.Tensor): A tensor containing all possible points where the acquisition function
                                        should be evaluated to determine the next suggested point.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
        batch_size (int): Number of candidates to generate.
        n_unique_features (int): Number of unique values of the constrained feature allowed
                                 in one given batch.
        feature_index (int): Index of this feature value in train_X and search_matrix.
        posterior_sampling (str): The method to use for sampling from the model's posterior. The default is 'TS_botorch'.

    Returns:
        torch.Tensor: Returns a batch of points selected by Bayesian Optimization.
    """
    # create a set containing number of unique temperatures
    unique_temps = set()
    non_naive_evaluations = 0

    X_next = torch.empty((0, search_matrix.size(1)), **tkwargs)
    evaluation_matrix = search_matrix.clone()

    train_and_pending_Y = train_Y_std.detach().clone()

    while len(unique_temps) < n_unique_features:
        Y_samples = _sample_model_posterior(
            model=model,
            likelihood=likelihood,
            evaluation_matrix=evaluation_matrix,
            max_batch_samples=10000,
            posterior_sampling=posterior_sampling,
        )

        candidate, best_index = setup_TS_HVI_and_optimize(
            evaluation_matrix,
            train_Y_std,
            train_and_pending_Y,
            Y_samples,
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        train_and_pending_Y = torch.cat(
            (train_and_pending_Y, Y_samples[best_index, :].unsqueeze(0)), dim=0
        )

        # append the temperature of the best candidate to the list of unique temperatures and increase counter on naive_evaluations
        unique_temps.add(
            evaluation_matrix[best_index, :][feature_index].item()
        )  # best_cand is just a one row tensor, so can directly use feature index
        non_naive_evaluations += 1

        # print(f"Current unique temperatures sampled: {unique_temps}")
        # print(f"Current unrestricted evaluation count: {non_naive_evaluations}")

        # remove the best candidate from the evaluation matrix
        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

        # Explicit memory cleanup after generation of each batch point
        gc.collect()
        torch.cuda.empty_cache()

        if non_naive_evaluations == batch_size:
            # print("Unrestricted evaluations completed. Batch size limit reached.")
            break

    # filter evaluation matrix to only include rows with the unique temperatures
    temperature_mask = torch.tensor(
        [row[feature_index].item() in unique_temps for row in evaluation_matrix]
    )
    evaluation_matrix = evaluation_matrix[temperature_mask]

    batch_budget = batch_size - non_naive_evaluations
    # print(f"remaining batch budget after unrestricted evaluations: {batch_budget}")

    naive_evaluations = 0

    while batch_budget > 0:
        Y_samples = _sample_model_posterior(
            model=model,
            likelihood=likelihood,
            evaluation_matrix=evaluation_matrix,
            max_batch_samples=10000,
            posterior_sampling="TS_botorch",
        )

        candidate, best_index = setup_TS_HVI_and_optimize(
            evaluation_matrix,
            train_Y_std,
            train_and_pending_Y,
            Y_samples,
        )

        X_next = torch.cat((X_next, candidate), dim=0)
        train_and_pending_Y = torch.cat(
            (train_and_pending_Y, Y_samples[best_index, :].unsqueeze(0)), dim=0
        )

        batch_budget -= 1
        naive_evaluations += 1

        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )
    # uncomment for logging

    # print("Batch generation completed.")
    # print(f"Total unrestricted evaluations: {non_naive_evaluations}")
    # print(f"Total restricted evaluations: {naive_evaluations}")
    # print(f"Unique temperatures selected: {unique_temps}")

    return X_next


def select_constrained_feature(
    model: ModelListGP,
    search_matrix: torch.Tensor,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    n_unique_features: int,
    feature_index: int,
    sampler: MCSampler,
    aggregation_strategy="mean",
) -> Tuple[torch.Tensor, List]:
    """
    In a given iteration of BO, we use the current state of the GP model trained
    on train_X and train_Y_std, and calculating the qNEHVI acquisition function
    value over the entire search space search_matrix using the worst point in both
    objectives as the reference point.

    The different possible values of the constrained feature, usually temperature,
    is cast in a dictionary, with all the acquisition function values of the
    reaction conditions with that feature.

    Different aggregations of the acquisition function value are used to select
    the most 'promising' constrained feature(s) to continue with, based on
    n_unique_features, and aggreagation_strategy.

    The search space, search_matrix, is filtered to return only the 'experiments'
    with the most promising feature(s).

    Args:
        model (ModelListGP): The trained GP model object from BoTorch
        search_matrix (torch.Tensor): Complete search space to evaluate acqf.
        train_X (torch.Tensor): Normalised input data of training data for GP
        train_Y_std (torch.Tensor): Standardised output/target training data for GP
        n_unique_features (int): Number of unique features allowed per batch
        feature_index (int): Index of feature in search_matrix to select from/restrict
        sampler (MCSampler): Sampler used to for MC Acquisition function.
        aggregation_strategy (str): Strategy used to aggregate feature acqf values

    Returns:
        Tuple[torch.Tensor, List]: Tensor of filtered search matrix and list of top temperatures.
    """

    # set ref point to calculate qNEHVI
    ref_point, ref_index = torch.min(train_Y_std, axis=0)

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        sampler=sampler,
        ref_point=ref_point,
        incremental_nehvi=True,
        prune_baseline=True,
        X_baseline=train_X,
    )

    # compute qNEHVI over the entire search space in batches
    with torch.no_grad():
        hvi = _split_batch_eval_acqf(
            acq_func, search_matrix.unsqueeze(-2), max_batch_size=2048
        )

    temp_values = (
        search_matrix[:, feature_index].detach().cpu().numpy()
    )  # extract temperatures from search matrix
    hvi_values = (
        hvi.detach().cpu().numpy()
    )  # extract hypervolume improvement values from acqf, convert to cpu if necessary

    top_temperatures = get_top_temperatures(
        temp_values=temp_values,
        acqf_values=hvi_values,
        n_temperatures=n_unique_features,
        aggregation_strategy=aggregation_strategy,
    )

    # filter the search matrix to include only the rows with the selected temperatures
    promising_temperature_experiments = torch.stack(
        [
            experiment
            for experiment in search_matrix
            if experiment[feature_index] in top_temperatures
        ]
    )

    return promising_temperature_experiments.to(train_X.device), top_temperatures


def aggregate_acqf_values(temp_dict: Dict, aggregation_strategy: str = "mean") -> Dict:
    """
    Aggregates the acquisition function values for each temperature based on the specified strategy.

    Args:
        temp_dict (dict): Dictionary containing temperature-wise acquisition function values.
        strategy (str): Strategy to use for aggregation. Options: 'mean', 'max', 'min'.

    Returns:
        Dict: Dictionary containing the aggregated acquisition function value for each temperature.
    """
    if aggregation_strategy not in ["mean", "max", "min"]:
        raise ValueError(f"Invalid aggregation strategy: {aggregation_strategy}")

    aggregated_dict = {}

    for temp in temp_dict:
        values = temp_dict[temp]["values"]

        if aggregation_strategy == "mean":
            aggregated_value = sum(values) / len(values)

        elif aggregation_strategy == "max":
            aggregated_value = max(values)

        elif aggregation_strategy == "min":
            aggregated_value = min(values)

        aggregated_dict[temp] = aggregated_value

    return aggregated_dict


def nested_acquisition_function(
    model: ModelListGP,
    likelihood: List[Likelihood],
    search_matrix: torch.Tensor,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    batch_size: int,
    n_unique_features: int,
    feature_index: int,
    search_strategy: str,
    aggregation_strategy: str = "mean",
    sampler: MCSampler = None,
    max_batch_eval: int = 256,
    nested_sampler: MCSampler = None,
) -> torch.Tensor:
    """
    Performs a two-stage nested acquisition function process to select the next set of experiments
    to run, based on the performance of specific features (e.g., temperatures) within a search matrix.

    The process consists of:
        1. Identifying the top-performing feature values (e.g., temperatures) based on the acqf values per
           for the experiments in each temperature. The aggregation strategy can be set, mean is used in the manuscript.
        2. Filtering the search space to include only experiments with these top-performing feature values.
        3. Re-running the acquisition function on this filtered search space to select the next experiments.

    Args:
        model (ModelListGP): A trained GP model object from BoTorch.
        likelihood (List[Likelihood]): The likelihood object corresponding to the GP model.
        search_matrix (torch.Tensor): The complete search space to evaluate the acquisition function.
        train_X (torch.Tensor): The normalized input data used for training the GP model.
        train_Y_std (torch.Tensor): The standardized output/target data used for training the GP model.
        batch_size (int): The number of experiments to select in each batch.
        n_unique_features (int): The number of unique feature values considered per batch.
        feature_index (int): The index of the feature in the search matrix to select from/restrict.
        search_strategy (str): The search strategy to use for optimizing the acquisition function.
            Supported strategies: 'TS-HVI', 'qNEHVI', 'qNParEGO'.
        aggregation_strategy (str, optional): The strategy used to aggregate feature acquisition function values.
            Defaults to 'mean'.
        sampler (MCSampler): The sampler object used for MC Acquisition function
        max_batch_eval (int): The maximum batch size for evaluating the acquisition function.
            Defaults to 256.
        nested_sampler (MCSampler): A separate sampler object to be used specifically within
            the nested acquisition function for selecting the most promising features.

    Returns:
        torch.Tensor: The selected experiments based on the nested acquisition function process.
    """
    # First, filter search matrix for top temperatures.
    filtered_search_matrix, top_temperatures = select_constrained_feature(
        model,
        search_matrix,
        train_X,
        train_Y_std,
        n_unique_features,
        feature_index,
        nested_sampler,
        aggregation_strategy,
    )

    # BO is run as normal using the filtered matrix.
    if search_strategy == "TS-HVI":
        new_x = optimize_TS_HVI_and_get_suggestion(
            model, likelihood, filtered_search_matrix, train_Y_std, batch_size
        )
    else:
        new_x = optimize_acqf_and_get_suggestion(
            model,
            search_strategy,
            train_X,
            train_Y_std,
            filtered_search_matrix,
            sampler,
            batch_size,
            max_batch_eval,
        )

    return new_x


def get_neg_utopia_distance(
    model: ModelListGP,
    search_matrix: torch.Tensor,
    std_scaler: StandardScaler,
) -> np.array:
    """
    From a trained model and the search matrix, for all points in the search matrix,
    compute posterior predicted means, and calculated negative distance from utopia point.

    Part of greedy search strategy where reactions are only evaluated based
    on the proximity of their predicted mean to the utopia point.

    Note:
        This function acts as an acquisition function, providing negative utopia
        distances as values which should be maximised for point selection.

    Args:
        model (ModelListGP): A trained GP model object from BoTorch
        search_matrix (torch.Tensor): Search space to evaluate utopia point distance.
        std_scaler (StandardScaler): scikit-learn scaler used to scale the train_y labels

    Returns:
        np.array: Negative utopia point distances for all reactions in search space.
    """
    # get predicted y values
    pred_y_means = model.posterior(search_matrix).mean.detach().cpu().numpy()
    pred_y_means = std_scaler.inverse_transform(pred_y_means)

    # set utopia point, can be changed for prioritise yield/selectivity
    utopia_point = torch.tensor([1.1, 1.1])

    # calculate utopia distances for each point with pred_means
    utopia_distances = torch.norm(torch.tensor(pred_y_means) - utopia_point, dim=1)

    return -utopia_distances.detach().cpu().numpy()


def get_top_temperatures(
    temp_values: np.array,
    acqf_values: np.array,
    n_temperatures: int,
    aggregation_strategy: str,
) -> List:
    """
    From computed acquisition function values for all points in the search matrix,
    aggregate into dictionaries and obtain the top_n_temperatures based on the
    acqusition function values or their experiments.

    Note:
        This assumes that maximising acquisition function value is the desired goal.

    Args:
        temp_values (np.array): List of temperatures for all reactions in search space
        acqf_values (np.array): List of acquisition function values for all reactions
                                in the search space
        n_temperatures (int): Number of unique temperatures to return
        aggregation_strategy (str): Whether to return temperatures sorted by max, mean, or min values.

    Output:
        List: List of top temperatures based on aggregation strategy.
    """

    temp_dict = {}
    for temp, hvi_value in zip(temp_values, acqf_values):
        if temp not in temp_dict:
            temp_dict[temp] = {"values": []}
        temp_dict[temp]["values"].append(hvi_value)

    aggregated_dict = aggregate_acqf_values(temp_dict, aggregation_strategy)
    sorted_temperatures = sorted(
        aggregated_dict.items(), key=lambda x: x[1], reverse=True
    )

    # get the top performing temperatures
    top_temperatures = [temp for temp, _ in sorted_temperatures[:n_temperatures]]

    return top_temperatures
