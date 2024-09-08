"""Acquisition function module."""

import gc
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import MCSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

from minerva.utils import get_index_from_lookup

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def optimize_acqf_and_get_suggestion(
    model: ModelListGP,
    acqf: str,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    search_matrix: torch.Tensor,
    sampler: MCSampler,
    batch_size: int,
    max_batch_eval: int = 256,  # default for botorch
) -> torch.Tensor:
    """
    Given a trained model and search space, runs one iteration of Bayesian Optimization.

    Args:
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        acqf (str): The acquisition function strategy to use, currently handles qNParEgo and qNEHVI
        train_X (torch.Tensor): A tensor of normalised training input data that has already been used to train the model.
                                Used as the baseline for comparison in the acquisition function.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
                                    Used to determine the reference point for hypervolume calculation.
        search_matrix (torch.Tensor): A tensor containing all possible points in the search_space.
        sampler (MCSampler): A BoTorch MCSampler for MC Acquisition function.
        batch_size (int): Number of candidates to generate.
        max_batch_eval (int): Batch size to compute the acquisition function with.

    Returns:
        torch.Tensor: Returns a batch of points selected by Bayesian Optimization.
    """
    # empty tensor to store new candidates
    X_next = torch.empty((0, search_matrix.size(1)), **tkwargs)
    evaluation_matrix = search_matrix.detach().clone()

    if acqf == "qNParEgo":
        with torch.no_grad():
            pred = model.posterior(train_X).mean
    else:
        pred = None

    for i in tqdm(range(batch_size), desc="Optimizing batch"):
        candidate, best_index = _get_next_candidate_and_index(
            acqf=acqf,
            pred=pred,
            model=model,
            train_X=train_X,
            train_Y_std=train_Y_std,
            sampler=sampler,
            X_next=X_next,
            evaluation_matrix=evaluation_matrix,
            max_batch_eval=max_batch_eval,
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        # remove best candidate from the current evaluation matrix
        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

        gc.collect()
        torch.cuda.empty_cache()

    return X_next


def naive_optimize_acqf_and_get_suggestion(
    model: ModelListGP,
    acqf: str,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    search_matrix: torch.Tensor,
    sampler: MCSampler,
    batch_size: int,
    n_unique_features: int,
    feature_index: int,
    max_batch_eval: int = 256,
) -> torch.Tensor:
    """
    Given a trained model and search space, runs one naive constrained iteration of Bayesian Optimization.

    Args:
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        acqf (str): The acquisition function strategy to use, currently handles qNParEgo and qNEHVI
        train_X (torch.Tensor): A tensor of normalised training input data that has already been used to train the model.
                                Used as the baseline for comparison in the acquisition function.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
                                    Used to determine the reference point for hypervolume calculation.
        search_matrix (torch.Tensor): A tensor containing all possible points in the search_space.
        sampler (MCSampler): A BoTorch MCSampler used for MC Acquisition function.
        batch_size (int): Number of candidates to generate.
        n_unique_features (int): Number of unique values of the constrained feature allowed
                                 in one given batch.
        feature_index (int): Index of this feature value in train_X and search_matrix.
        max_batch_eval (int): Batch size to compute the acquisition function with.

    Returns:
        torch.Tensor: Returns a batch of points selected by Bayesian Optimization.
    """

    unique_temps = set()

    # log number of evaluations used to create the set of unique temperatures
    non_naive_evaluations = 0

    X_next = torch.empty((0, search_matrix.size(1)), **tkwargs)

    evaluation_matrix = search_matrix.clone()

    if acqf == "qNParEgo":
        with torch.no_grad():
            pred = model.posterior(train_X).mean
    else:
        pred = None

    while len(unique_temps) < n_unique_features:
        candidate, best_index = _get_next_candidate_and_index(
            acqf=acqf,
            pred=pred,
            model=model,
            train_X=train_X,
            train_Y_std=train_Y_std,
            sampler=sampler,
            X_next=X_next,
            evaluation_matrix=evaluation_matrix,
            max_batch_eval=max_batch_eval,
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        # append the temperature of the best candidate to the list of unique temperatures and increase counter on naive_evaluations
        unique_temps.add(evaluation_matrix[best_index, :][feature_index].item())
        non_naive_evaluations += 1

        # print(f"Current unique temperatures sampled: {unique_temps}")
        # print(f"Current unrestricted evaluation count: {non_naive_evaluations}")

        # remove best candidate from the current evaluation matrix, after adding the temperature to the set of unique temperatures
        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

        gc.collect()
        torch.cuda.empty_cache()

        if non_naive_evaluations == batch_size:
            # print("Unrestricted evaluations completed. Batch size limit reached. ")
            break

    # filter evaluation matrix to only include rows with the unique temperatures
    temperature_mask = torch.tensor(
        [row[feature_index].item() in unique_temps for row in evaluation_matrix]
    )

    # this matrix has removed selected points and also filtered by temp.
    evaluation_matrix = evaluation_matrix[temperature_mask]

    batch_budget = batch_size - non_naive_evaluations
    # print(f"remaining batch budget after unrestricted evaluations : {batch_budget}")

    naive_evaluations = 0

    while batch_budget > 0:
        candidate, best_index = _get_next_candidate_and_index(
            acqf=acqf,
            pred=pred,
            model=model,
            train_X=train_X,
            train_Y_std=train_Y_std,
            sampler=sampler,
            X_next=X_next,
            evaluation_matrix=evaluation_matrix,
            max_batch_eval=max_batch_eval,
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        batch_budget -= 1
        naive_evaluations += 1

        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

        gc.collect()
        torch.cuda.empty_cache()

    # uncomment for logging statements

    # print("Batch generation completed.")
    # print(f"Total unrestricted evaluations: {non_naive_evaluations}")
    # print(f"Total restricted evaluations: {naive_evaluations}")
    # print(f"Unique temperatures selected: {unique_temps}")

    return X_next


def _get_next_candidate_and_index(
    acqf: str,
    pred: torch.Tensor,
    model: ModelListGP,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    sampler: MCSampler,
    X_next: torch.Tensor,
    evaluation_matrix: torch.Tensor,
    max_batch_eval: int,
) -> Tuple[torch.Tensor, int]:
    """
    Get the next suggested candidate point and its index in evaluation_matrix.

    Args:
        acqf (str): The acquisition function strategy to use, currently handles qNParEgo and qNEHVI
        pred (torch.Tensor): Predicted means from the GP model's posterior over the training data.
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        train_X (torch.Tensor): A tensor of normalised training input data that has already been used to train the model.
                                Used as the baseline for comparison in the acquisition function.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
                                    Used to determine the reference point for hypervolume calculation.
        sampler (MCSampler): A BoTorch MCSampler for MC Acquisition function.
        X_next (torch.Tensor): A tensor of pending evaluation points.
        evaluation_matrix (torch.Tensor): Tensor containing all possible points for evaluating the acquisition function.
        max_batch_eval (int): Maximum number of evaluations that the optimizer will process in one batch.

    Returns:
        Tuple[torch.Tensor, int]: The single best candidate point from the evaluation matrix as
                                  determined by optimizing the acquisition function and its index.
    """

    if acqf == "qNParEgo":
        return setup_qnparego_and_optimize(
            pred,
            model,
            train_X,
            train_Y_std,
            sampler,
            evaluation_matrix,
            max_batch_eval,
        )
    elif acqf == "qNEHVI":
        return setup_qnehvi_and_optimize(
            model,
            sampler,
            train_X,
            train_Y_std,
            X_next,
            evaluation_matrix,
            max_batch_eval,
        )
    else:
        raise ValueError(f"Unsupported acquisition function type: {acqf}")


def setup_qnehvi_and_optimize(
    model: ModelListGP,
    sampler: MCSampler,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    X_next: torch.Tensor,
    evaluation_matrix: torch.Tensor,
    max_batch_eval: int,
) -> Tuple[torch.Tensor, int]:
    """
    Set up and optimize the qNEHVI acquisition function for Bayesian optimization.

    This function initializes the acquisition function using the provided model and samples,
    and performs optimization over a given evaluation matrix to find the best candidate point.

    Args:
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        sampler (MCSampler): A BoTorch MCSampler used for MC Acquisition function.
        train_X (torch.Tensor): A tensor of normalised training input data that has already been used to train the model.
                                Used as the baseline for comparison in the acquisition function.
        train_Y_std (torch.Tensor): A tensor of standardized training output data corresponding to `train_X`.
                                    Used to determine the reference point for hypervolume calculation.
        X_next (torch.Tensor): A tensor of pending evaluation points.
        evaluation_matrix (torch.Tensor): Tensor containing all possible points for evaluating the acquisition function.
        max_batch_eval (int): Maximum number of evaluations that the optimizer will process in one batch.

    Returns:
        Tuple[torch.Tensor, int]: The single best candidate point from the evaluation matrix as
                                  determined by optimizing the acquisition function and its index.
    """
    ref_point, ref_index = torch.min(
        train_Y_std, axis=0
    )  # ref point taken to be worst in both objectives.

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        sampler=sampler,
        ref_point=ref_point,
        incremental_nehvi=True,
        prune_baseline=True,
        X_baseline=train_X,
        X_pending=X_next,
    )

    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=evaluation_matrix,
        max_batch_size=max_batch_eval,
        q=1,
    )

    # get index of best candidate from the current evaluation matrix
    best_index = get_index_from_lookup(candidate, evaluation_matrix)
    best_index = best_index[0]  # for when we get a list back

    return candidate, best_index


def setup_qnparego_and_optimize(
    pred: torch.Tensor,
    model: ModelListGP,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    sampler: MCSampler,
    evaluation_matrix: torch.Tensor,
    max_batch_eval: int,
) -> Tuple[torch.Tensor, int]:
    """
    Set up and optimize qNParEgo for Bayesian Optimization.

    This function samples weights from a simplex to create a scalarized objective for each batch point,
    then uses these objectives to guide the Bayesian optimization process with the qNoisyExpectedImprovement.

    Args:
        pred (torch.Tensor): Predicted means from the GP model's posterior over the training data.
        model (ModelListGP): A Botorch Gaussian Process model fitted on training data.
        train_X (torch.Tensor): A tensor of normalised training input data that has already been used to train the model.
                                Used as the baseline for comparison in the acquisition function.
        train_Y_std (torch.Tensor): Tensor containing the standardized training output data.
        sampler (MCSampler): A BoTorch MCSampler used for MC Acquisition function.
        evaluation_matrix (torch.Tensor): Tensor containing all possible points for evaluating the acquisition function.
        max_batch_eval (int): Maximum number of evaluations that the optimizer will process in one batch.

    Returns:
        Tuple[torch.Tensor, int]: The single best candidate point from the evaluation matrix as
                                  determined by optimizing the acquisition function and its index.
    """
    # sample weights uniformly from a d-simplex to create a scalarised objective which is different for every batch point
    weights = sample_simplex(d=train_Y_std.size(1), **tkwargs).squeeze()
    objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))

    acq_func = qNoisyExpectedImprovement(
        model=model,
        objective=objective,
        X_baseline=train_X,
        sampler=sampler,
        prune_baseline=True,
    )

    # Evaluate acquisition function over the discrete search matrix
    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=evaluation_matrix,
        max_batch_size=max_batch_eval,
        q=1,
    )

    # get index of best candidate from the current evaluation matrix
    best_index = get_index_from_lookup(candidate, evaluation_matrix)
    best_index = best_index[0]  # for when we get a list back

    return candidate, best_index
