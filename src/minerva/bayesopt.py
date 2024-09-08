"Class for running ML optimisation"

import time
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import gpytorch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from botorch.sampling.samplers import SobolQMCNormalSampler

from minerva.gp_model import initialize_gp_model
from minerva.acqfunc import (
    optimize_acqf_and_get_suggestion,
    naive_optimize_acqf_and_get_suggestion,
)
from minerva.ts_acqfunc import (
    nested_acquisition_function,
    optimize_TS_HVI_and_get_suggestion,
    naive_optimize_TS_HVI_and_get_suggestion,
    get_neg_utopia_distance,
    aggregate_acqf_values,
    get_top_temperatures,
)
from minerva.utils import (
    count_identified_pareto_points,
    draw_sobol_samples,
    compute_hypervolume,
    get_pareto_points,
    calculate_igd,
    calculate_igd_plus,
    filter_tensor_random,
    unified_objective_function,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class BayesianOptimisation:
    """
    Class to run Bayesian optimisation benchmarks on existing datasets.
    """

    def __init__(self, device: Dict, seed: int, suppress_warnings: bool = True):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialising BayesianOptimisation with seed {seed}")

        self.device = device
        self.seed = seed
        self.scaler_features = MinMaxScaler()

        self.data_df = None
        self.objective_columns = None

        self.suppress_warnings = suppress_warnings
        if self.suppress_warnings:
            warnings.filterwarnings("ignore")

        pl.seed_everything(seed)

    def set_fast_computation(self):
        """
        Set fast computation settings for GPyTorch.
        Reccomended for higher batch sizes and larger search spaces.

        Note:
        - This will change optimisation results slightly.
        - Baselines were not run with this setting.
        """

        gpytorch.settings.fast_pred_var._state = True
        gpytorch.settings.fast_pred_samples._state = True
        gpytorch.settings.fast_computations.covar_root_decomposition._state = True
        gpytorch.settings.fast_computations.log_prob._state = True
        gpytorch.settings.fast_computations.solves._state = True
        gpytorch.settings.memory_efficient._state = True

    def load_and_preprocess_benchmark(
        self, data_df_path: str, objective_columns: List[str]
    ):
        """
        Loads data from a csv file and preprocesses it for optimisation.

        This method constructs x_main and y_main search space tensors.
        This method also calculates the pareto front and optimal hypervolume of
        benchmark ground truth.
        """
        self.logger.info(
            f"Loading data from {data_df_path} with objective columns {objective_columns}"
        )

        self.load_df(data_df_path, objective_columns)
        self.set_feature_space()
        self.get_optimal_pareto_front()
        self.get_optimal_hypervolume()

    def load_df(self, data_df_path: str, objective_columns: List[str]):
        """
        Loads dataframe/benchmark csv.

        Inputs
        - data_df_path (str): path from which to read dataframe from
        - objective_columns (List): columns corresponding to objective function values in the benchmark dataframe
        """

        df = pd.read_csv(data_df_path)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

        self.df = df
        self.objective_columns = objective_columns

    def set_feature_space(self):
        "Convert loaded data to tensors and scale features, storing in self.x_main and self.y_main"
        self.logger.info("Normalising input features and converting to torch tensors.")

        columns_regression = self.df.columns
        columns_regression = columns_regression.drop(self.objective_columns).tolist()

        df_features = self.df[columns_regression]
        self.scaler_features.fit(df_features.to_numpy())
        x_features = self.scaler_features.transform(df_features.to_numpy())
        x_main = torch.tensor(x_features).to(**self.device)

        df_objectives = self.df[self.objective_columns]
        y_objectives = df_objectives.astype(float).to_numpy()
        y_main = torch.tensor(y_objectives).to(**self.device)

        self.x_main = x_main
        self.y_main = y_main

    def get_train_x_and_y(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get initial train x and y from drawing sobol samples from defined search space

        Note:
        - Code is designed to run in order, after the feature space is already set.
        - Objective function is a lookup function for the corresponding X index in the Y column to retrieve values.
        """

        train_x = draw_sobol_samples(n_samples=n_samples, feature_matrix=self.x_main)
        train_y = unified_objective_function(train_x, self.x_main, self.y_main)

        return train_x, train_y

    def get_ref_point(self) -> torch.Tensor:
        "Get reference point for hypervolume calculation. Use global worst in both objectives."

        global_ref, global_ref_index = torch.min(self.y_main, axis=0)

        return global_ref

    def get_optimal_pareto_front(self):
        "Get optimal pareto front from ground truth Y"
        self.logger.info("Getting pareto front of benchmark dataset")

        gt_pareto_points = get_pareto_points(self.y_main)

        self.gt_pareto_points = gt_pareto_points

    def get_optimal_hypervolume(self):
        """
        Calculate optimal hypervolume from all Y data using the ref (min in  both objectives) point.

        This is used to compare campaign hypervolumes against
        """
        self.logger.info(
            "Calculating optimal hypervolume of benchmark dataset as reference."
        )

        ref_point = self.get_ref_point()
        hv_max = compute_hypervolume(ref_point, self.y_main)

        self.hv_max = hv_max

    def _initialize_metrics(self, train_y: torch.Tensor) -> Tuple:
        """
        Initialize metrics for hypervolume and IGD+ at the start of the optimization campaign.

        Args:
            train_y (torch.Tensor): Initial training output data.

        Returns:
            Tuple: Contains the ref point, initial hypervolume, hypervolume log, initial IGD+, and IGD+ log.
        """
        ref_point = self.get_ref_point()
        init_volume = compute_hypervolume(ref_point, train_y)
        hv_log = [init_volume]

        init_igd_plus = calculate_igd_plus(
            pf_true=self.gt_pareto_points.cpu().numpy(),
            pf_approx=get_pareto_points(train_y).cpu().numpy(),
        )
        igd_plus_log = [init_igd_plus]

        return ref_point, init_volume, hv_log, init_igd_plus, igd_plus_log

    def _log_iteration_results(
        self,
        ref_point: torch.Tensor,
        train_y: torch.Tensor,
        hv_log: List,
        igd_plus_log: List,
    ):
        """
        Logs results of each iteration, such as hypervolume and IGD+.

        Args:
            ref_point (torch.Tensor): The reference point used for hypervolume calculations.
            train_y (torch.Tensor): Updated training outputs after adding new observations.
            hv_log (list): Log of hypervolume values across iterations.
            igd_plus_log (list): Log of IGD+ values across iterations.
        """
        volume = compute_hypervolume(ref_point, train_y)
        hv_log.append(volume)
        igd_plus = calculate_igd_plus(
            pf_true=self.gt_pareto_points.cpu().numpy(),
            pf_approx=get_pareto_points(train_y).cpu().numpy(),
        )
        igd_plus_log.append(igd_plus)

    def _finalize_optimisation_campaign(
        self,
        start_time: float,
        train_y: torch.Tensor,
        batch_size: int,
        n_iterations: int,
        hv_log: List,
        igd_plus_log: List,
        search_strategy: str,
        kernel: str,
        acqf_samples: int,
        init_samples: int,
        use_keops: bool,
    ) -> Dict:
        """
        Finalizes the optimization run by computing metrics and compiling results.
        """

        end_time = time.time()
        elapsed_time = end_time - start_time
        campaign_pareto = get_pareto_points(train_y)
        pareto_points_identified = count_identified_pareto_points(
            self.gt_pareto_points, campaign_pareto, tolerance=0
        )

        # print statements to terminal
        print(
            f"Time taken for optimisation run of batch size {batch_size} and {n_iterations} iterations: {elapsed_time} seconds"
        )
        print(f"The optimal hypervolume is: {self.hv_max}")
        print(
            f"Hypervolume log: {hv_log} and max hypervolume achieved in campaign: {np.max(hv_log)}"
        )
        print(
            f"The number of ground truth pareto points identified: {pareto_points_identified} out of {len(self.gt_pareto_points.cpu().numpy())}"
        )
        print(
            f"The min inverted generational distance is {calculate_igd(pf_true=self.gt_pareto_points.cpu().numpy(), pf_approx=campaign_pareto.cpu().numpy())}"
        )
        print(f"IGD plus log: {igd_plus_log}")

        results = {
            "Search Strategy": search_strategy,
            "Kernel": kernel,
            "acqf_samples": acqf_samples,
            "initial_samples": init_samples,
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "Use KeOps": use_keops,
            "Time": elapsed_time,
            "Pareto Points Identified": pareto_points_identified,
            "Total ground truth pareto points": len(
                self.gt_pareto_points.cpu().numpy()
            ),
            "IGD": calculate_igd(
                pf_true=self.gt_pareto_points.cpu().numpy(),
                pf_approx=campaign_pareto.cpu().numpy(),
            ),
            "Optimal Hypervolume": self.hv_max,
        }

        # Log hypervolume and IGD+ per iteration
        for i, hv in enumerate(hv_log):
            results[f"Hypervolume Iteration {i}"] = hv
        for i, igd_plus in enumerate(igd_plus_log):
            results[f"IGD+ Iteration {i}"] = igd_plus

        return results

    def run_baseline_benchmark(
        self,
        init_samples: int,
        batch_size: int,
        n_iterations: int,
        search_strategy: str,
        acqf_samples: int = 512,
        use_keops: bool = False,
        max_batch_eval: int = 256,
        kernel: str = None,
        noise_std: float = 0,
        noise_std_2: float = 0,
    ):
        """
        Run a standard bayesian optimisation with the same settings and search strategy for all iterations.
        Metrics are printed to the terminal.

        Optionally, the observations used to train the GP model are permuted with Gaussian noise
        of specified noise standard deviation.

        Args:
            init_samples (int): Number of initial samples to draw using sobol sampling.
            batch_size (int): Number of candidates to generate per iteration.
            n_iterations (int): Number of iterations to run the optimisation.
            search_strategy (str): Search strategy to use for the optimisation, TS_HVI, qNParEgo, or qNEHVI
            use_keops (bool): Whether to use KeOps for the GP kernel, default is False.
                              Will speed up computation but results will change.
            max_batch_eval (int): batch size for acquisition function evaluation, may want to reduce for memory issues
            kernel (str): Choice of kernel/GP to use as a surrogate model
            noise_std (int): Gaussian noise std used to permute the first objective
            noise_std_2 (int): Gaussian noise std used to permute the second objective

        Optional Args only required for qNParEgo and qNEHVI:
            acqf_samples (int): Number of MC Samples used for MC acquisition function.
        """
        start_time = time.time()

        # initialise training data and labels
        train_x, train_y = self.get_train_x_and_y(n_samples=init_samples)

        # noise free train_y used to calculate campaign metrics
        ref_point, init_volume, hv_log, init_igd_plus, igd_plus_log = (
            self._initialize_metrics(train_y)
        )

        # noisy tran_y used as training data
        noisy_train_y = unified_objective_function(
            train_x, self.x_main, self.y_main, noise_stds=[noise_std, noise_std_2]
        )

        if search_strategy in ["qNParEgo", "qNEHVI"]:
            sampler = SobolQMCNormalSampler(num_samples=acqf_samples, seed=self.seed)
        else:
            sampler = None

        for i in tqdm(range(1, n_iterations + 1)):
            # scale training labels for gp model training
            scaler_y = StandardScaler()
            train_y_std = noisy_train_y.detach().clone()
            train_y_std = scaler_y.fit_transform(train_y_std.detach().cpu().numpy())
            train_y_std = torch.tensor(train_y_std).to(**self.device)

            if kernel is not None:
                model, likelihood = initialize_gp_model(
                    train_x, train_y_std, use_keops=use_keops, kernel=kernel
                )

            with torch.no_grad():
                if search_strategy == "sobol":
                    new_x = draw_sobol_samples(
                        n_samples=batch_size, feature_matrix=self.x_main
                    )
                elif search_strategy == "TS-HVI":
                    new_x = optimize_TS_HVI_and_get_suggestion(
                        model,
                        likelihood,
                        self.x_main,
                        train_y_std,
                        batch_size,
                        posterior_sampling="TS_botorch",
                    )
                else:
                    new_x = optimize_acqf_and_get_suggestion(
                        model,
                        search_strategy,
                        train_x,
                        train_y_std,
                        self.x_main,
                        sampler,
                        batch_size,
                        max_batch_eval=max_batch_eval,
                    )

            # obtain unscaled new_y to calculate and log metrics
            new_y = unified_objective_function(new_x, self.x_main, self.y_main)
            train_y = torch.cat([train_y, new_y])

            # extract noisy objective function values for new sampled points
            new_noisy_train_y = unified_objective_function(
                new_x, self.x_main, self.y_main, noise_stds=[noise_std, noise_std_2]
            )

            # append new 'training data', this time with noisy observations
            train_x = torch.cat([train_x, new_x])
            noisy_train_y = torch.cat([noisy_train_y, new_noisy_train_y])

            # update metrics
            self._log_iteration_results(ref_point, train_y, hv_log, igd_plus_log)

        # log and return all metrics
        campaign_run = {"Noise Std": noise_std, "Noise Std 2": noise_std_2}

        campaign_results = self._finalize_optimisation_campaign(
            start_time,
            train_y,
            batch_size,
            n_iterations,
            hv_log,
            igd_plus_log,
            search_strategy,
            kernel,
            acqf_samples,
            init_samples,
            use_keops,
        )

        campaign_run.update(campaign_results)

        return campaign_run

    def run_constrained_benchmark(
        self,
        init_samples: int,
        batch_size: int,
        n_iterations: int,
        search_strategy: str,
        constrain_strategy: str,
        n_unique_features: int,
        feature_index: int,
        acqf_samples: int = 512,
        use_keops: bool = False,
        max_batch_eval: int = 256,
        kernel: str = None,
        noise_std: float = 0,
        noise_std_2: float = 0,
    ):
        """
        Run a constrained bayesian optimisation where there are limits on batches.
        User may select constrain strategies. Metrics are printed to the terminal.

        Args:
            init_samples (int): Number of initial samples to draw using sobol sampling.
            batch_size (int): Number of candidates to generate per iteration.
            n_iterations (int): Number of iterations to run the optimisation.
            search_strategy (str): Search strategy to use for the optimisation, TS_HVI, qNParEgo, or qNEHVI
            constrain_strategy (str): Strategy to use for directing the acquisition function.
                                      Either naive or nested strategies.
            n_unique_features (int): Number of unique features allowed in the batch
            feature_index (int): Index of constrained feature
            use_keops (bool): Whether to use KeOps for the GP kernel, default is False.
                              Will speed up computation but results will change.
            max_batch_eval (int): batch size for acquisition function evaluation, may want to reduce for memory issues
            kernel (str): Choice of kernel/GP to use as a surrogate model
            noise_std (float): Standard deviation of noise to add to the first objective
            noise_std_2 (float): Standard deviation of noise to add to the second objective

        Optional Args only required for qNParEgo and qNEHVI:
            acqf_samples (int): Number of MC Samples used for MC acquisition function.
        """
        start_time = time.time()

        # create filtered matrix to initialise constrained BO.
        constrain_matrix, constrain_labels = filter_tensor_random(
            self.x_main,
            self.y_main,
            feature_index=feature_index,
            n_unique_features=n_unique_features,
        )

        # draw sobol samples from filtered matrix to initialise training data
        train_x = draw_sobol_samples(n_samples=init_samples, feature_matrix=constrain_matrix)
        train_y = unified_objective_function(train_x, self.x_main, self.y_main)

        # initialise metrics for logging.
        ref_point, init_volume, hv_log, init_igd_plus, igd_plus_log = (
            self._initialize_metrics(train_y)
        )

        noisy_train_y = unified_objective_function(
            train_x, self.x_main, self.y_main, noise_stds=[noise_std, noise_std_2]
        )

        if search_strategy in ["qNParEgo", "qNEHVI"]:
            sampler = SobolQMCNormalSampler(num_samples=acqf_samples, seed=self.seed)
        else:
            sampler = None

        nested_sampler = SobolQMCNormalSampler(num_samples=acqf_samples, seed=self.seed)

        for i in tqdm(range(1, n_iterations + 1)):
            # scale training labels for gp model training
            scaler_y = StandardScaler()
            train_y_std = noisy_train_y.detach().clone()
            train_y_std = scaler_y.fit_transform(train_y_std.detach().cpu().numpy())
            train_y_std = torch.tensor(train_y_std).to(**self.device)

            if kernel is not None:
                model, likelihood = initialize_gp_model(
                    train_x, train_y_std, use_keops=use_keops, kernel=kernel
                )

            with torch.no_grad():
                if constrain_strategy == "naive":
                    if search_strategy == "TS-HVI":
                        new_x = naive_optimize_TS_HVI_and_get_suggestion(
                            model,
                            likelihood,
                            self.x_main,
                            train_y_std,
                            batch_size,
                            n_unique_features=n_unique_features,
                            feature_index=feature_index,
                            posterior_sampling="TS_botorch",
                        )
                    else:
                        new_x = naive_optimize_acqf_and_get_suggestion(
                            model,
                            search_strategy,
                            train_x,
                            train_y_std,
                            self.x_main,
                            sampler,
                            batch_size,
                            n_unique_features=n_unique_features,
                            feature_index=feature_index,
                            max_batch_eval=max_batch_eval,
                        )

                if constrain_strategy == "nested":
                    new_x = nested_acquisition_function(
                        model,
                        likelihood,
                        search_matrix=self.x_main,
                        train_X=train_x,
                        train_Y_std=train_y_std,
                        batch_size=batch_size,
                        n_unique_features=n_unique_features,
                        feature_index=feature_index,
                        search_strategy=search_strategy,
                        sampler=sampler,
                        max_batch_eval=max_batch_eval,
                        nested_sampler=nested_sampler,
                    )

            # obtain unscaled new_y to calculate and log metrics
            new_y = unified_objective_function(new_x, self.x_main, self.y_main)
            train_y = torch.cat([train_y, new_y])

            # extract noisy objective function values for new sampled points
            new_noisy_train_y = unified_objective_function(
                new_x, self.x_main, self.y_main, noise_stds=[noise_std, noise_std_2]
            )

            # append new 'training data', with noisy observations
            train_x = torch.cat([train_x, new_x])
            noisy_train_y = torch.cat([noisy_train_y, new_noisy_train_y])

            # calculate hypervolume using worst global point as reference - no need for standardisation
            self._log_iteration_results(ref_point, train_y, hv_log, igd_plus_log)

        # log and return all metrics
        campaign_run = {
            "Constrain strategy": constrain_strategy,
            "Unique features per patch": n_unique_features,
        }

        campaign_results = self._finalize_optimisation_campaign(
            start_time,
            train_y,
            batch_size,
            n_iterations,
            hv_log,
            igd_plus_log,
            search_strategy,
            kernel,
            acqf_samples,
            init_samples,
            use_keops,
        )

        campaign_run.update(campaign_results)

        return campaign_run

    def run_optimisation_iteration(
        self,
        train_x: torch.Tensor,
        train_y_std: torch.Tensor,
        x_space: torch.Tensor,
        batch_size: int,
        search_strategy: str = "qNEHVI",
        kernel: str = "edboplus",
        use_keops: bool = False,
        acqf_samples: int = 256,
        max_batch_eval: int = 256,
        constrain_strategy: str = None,
        n_unique_features: int = None,
        feature_index: int = None,
    ) -> torch.Tensor:
        """
        Given some training data/initial results, and the desired discrete reaction condition
        search space to explore, run one iteration of Bayesian optimisation.

        Args:
            train_x (torch.Tensor): Normalised training features as tensor input to GP.
            train_y_std (torch.Tensor): Standardised training labels as tensor input to GP.
            x_space (torch.Tensor): Desired reaction condition space to explore as a tensor.
            batch_size (int): Number of candidates to generate per iteration.
            search_strategy (str): Search strategy to use for the optimisation, TS_HVI, qNParEgo, or qNEHVI
            kernel (str): Choice of kernel/GP to use as a surrogate model
            use_keops (bool): Whether to use KeOps for the GP kernel, default is False.
                              Will speed up computation but results will change.
            acqf_samples (int): Number of samples used for MC Acquisition function.
            max_batch_eval (int): batch size for acquisition function evaluation, may want to reduce for memory issues
            constrain_strategy (str): Strategy to use for directing the acquisition function.
                                      Either naive or nested strategies.
            n_unique_features (int): Number of unique features allowed in the batch
            feature_index (int): Index of constrained feature

        Returns:
            torch.Tensor: selected batch points
        """
        # train model
        model, likelihood = initialize_gp_model(
            train_x, train_y_std, use_keops=use_keops, kernel=kernel
        )

        if search_strategy in ["qNParEgo", "qNEHVI"]:
            sampler = SobolQMCNormalSampler(num_samples=acqf_samples, seed=self.seed)
        else:
            sampler = None

        with torch.no_grad():
            if constrain_strategy == "naive":
                if search_strategy == "TS-HVI":
                    new_x = naive_optimize_TS_HVI_and_get_suggestion(
                        model,
                        likelihood,
                        x_space,
                        train_y_std,
                        batch_size,
                        n_unique_features=n_unique_features,
                        feature_index=feature_index,
                        posterior_sampling="TS_botorch",
                    )
                else:
                    new_x = naive_optimize_acqf_and_get_suggestion(
                        model,
                        search_strategy,
                        train_x,
                        train_y_std,
                        x_space,
                        sampler,
                        batch_size,
                        n_unique_features=n_unique_features,
                        feature_index=feature_index,
                        max_batch_eval=max_batch_eval,
                    )
            elif constrain_strategy == "nested":
                nested_sampler = SobolQMCNormalSampler(
                    num_samples=acqf_samples, seed=self.seed
                )

                new_x = nested_acquisition_function(
                    model,
                    likelihood,
                    search_matrix=x_space,
                    train_X=train_x,
                    train_Y_std=train_y_std,
                    batch_size=batch_size,
                    n_unique_features=n_unique_features,
                    feature_index=feature_index,
                    search_strategy=search_strategy,
                    sampler=sampler,
                    max_batch_eval=max_batch_eval,
                    nested_sampler=nested_sampler,
                )
            elif constrain_strategy is None:
                if search_strategy == "TS-HVI":
                    new_x = optimize_TS_HVI_and_get_suggestion(
                        model,
                        likelihood,
                        x_space,
                        train_y_std,
                        batch_size,
                        posterior_sampling="TS_botorch",
                    )
                else:
                    new_x = optimize_acqf_and_get_suggestion(
                        model,
                        search_strategy,
                        train_x,
                        train_y_std,
                        x_space,
                        sampler,
                        batch_size,
                        max_batch_eval=max_batch_eval,
                    )

        return new_x

    def run_exploitative_iteration(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        x_space: torch.Tensor,
        batch_size: int,
        kernel: str,
        use_keops: bool = False,
        constrain: Optional[bool] = True,
        n_unique_features: Optional[int] = None,
        feature_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Given some training data/initial results, and the desired discrete reaction condition
        search space to explore, run one iteration of exploitative optimisation.

        Args:
            train_x (torch.Tensor): Normalised training features as tensor input to GP.
            train_y (torch.Tensor): Unscaled training labels to be scaled as input to GP.
            x_space (torch.Tensor): Desired reaction condition space to explore as a tensor. Normalised
            batch_size (int): Number of candidates to generate per iteration.
            kernel (str): Choice of kernel/GP to use as a surrogate model
            use_keops (bool): Whether to use KeOps for the GP kernel, default is False.
                              Will speed up computation but results will change.
            constrain (bool): Whether or not to constrain the batch selection
            n_unique_features (int): Number of unique features allowed in the batch
            feature_index (int): Index of constrained feature

        Returns:
            torch.Tensor: selected batch points

        Returns:

        """
        if constrain and (n_unique_features is None or feature_index is None):
            raise ValueError(
                "n_unique_features and feature_index must be provided when constrain is True"
            )

        # scale train_y
        scaler_objectives = StandardScaler()
        train_y_std = scaler_objectives.fit_transform(train_y)
        train_y_std = torch.tensor(train_y_std).to(train_x)

        # train model
        model, likelihood = initialize_gp_model(
            train_x, train_y_std, use_keops=use_keops, kernel=kernel
        )

        # get negative utopia distances for each point in search matrix
        full_acqf_values = get_neg_utopia_distance(
            model=model, search_matrix=x_space, std_scaler=scaler_objectives
        )

        if constrain is True:
            temp_values = x_space[:, feature_index].detach().cpu().numpy()

            # sort temperatures by max (negative) utopia distance
            top_temperatures = get_top_temperatures(
                temp_values=temp_values,
                acqf_values=full_acqf_values,
                n_temperatures=n_unique_features,
                aggregation_strategy="max",
            )

            # filter the search matrix to include only the rows with the selected temperatures
            promising_experiments = torch.stack(
                [
                    experiment
                    for experiment in x_space
                    if experiment[feature_index] in top_temperatures
                ]
            ).to(train_x.device)

            # get negative utopia distances for points in filtered matrix
            neg_utopia_distances = get_neg_utopia_distance(
                model=model,
                search_matrix=promising_experiments,
                std_scaler=scaler_objectives,
            )

        else:
            neg_utopia_distances = full_acqf_values
            promising_experiments = x_space

        # sort utopia distances from lowest to largest and take minimum
        best_indices = np.argsort(-neg_utopia_distances)[:batch_size]
        best_points = promising_experiments[best_indices]

        return best_points
