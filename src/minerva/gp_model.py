"""
Implementation of GPyTorch GP with kernels and code adapted from EDBO+ and MorBO
github repository: https://github.com/doyle-lab-ucla/edboplus, https://github.com/facebookresearch/morbo
"""

from typing import Optional, Dict, Any, List, Tuple

import torch
import gpytorch
import numpy as np

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.likelihoods import Likelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.kernels.keops import MaternKernel as KMaternKernel

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

gp_options = {
    "ls_prior1": 2.0,
    "ls_prior2": 0.2,
    "ls_prior3": 5.0,
    "out_prior1": 5.0,
    "out_prior2": 0.5,
    "out_prior3": 8.0,
    "noise_prior1": 1.5,
    "noise_prior2": 0.1,
    "noise_prior3": 5.0,
    "noise_constraint": 1e-5,
}


def initialize_gp_model(
    train_x: torch.Tensor,
    train_y_std: torch.Tensor,
    use_keops: bool = False,
    kernel: str = "default",
) -> Tuple[ModelListGP, List[Likelihood]]:
    """
    Helper function to initialize and returns a BoTorch compatible GP model
    and likelihood based on the specified kernel.

    Parameters:
    - train_x (torch.Tensor): The training input data (features), assumed to be normalised.
    - train_y_std (torch.Tensor): The standardized training output data (targets) assumed to be standardized.
    - use_keops (bool): A flag indicating whether to use KeOps for kernel computations.
    - kernel (str): The type of kernel to be used for the GP model. Options are 'edboplus' or 'default'.

    Returns:
    - model (ModelListGP): The initialized BoTorch compatible GP.
    - likelihood (List[Likelihood]): The likelihood associated with the GP model objectives.
    """
    if kernel == "edboplus":
        model, likelihood = get_fitted_gpytorch_gp(
            train_x, train_y_std, use_keops=use_keops
        )
    elif kernel == "default":
        model, likelihood = get_fitted_botorch_gp(train_x, train_y_std)

    return model, likelihood


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type=MaternKernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        n_features = np.shape(train_x)[1]

        self.mean_module = gpytorch.means.ConstantMean()

        kernels = kernel_type(
            ard_num_dims=n_features,
            lengthscale_prior=GammaPrior(
                gp_options["ls_prior1"], gp_options["ls_prior2"]
            ),
        )

        self.covar_module = ScaleKernel(
            kernels,
            outputscale_prior=GammaPrior(
                gp_options["out_prior1"], gp_options["out_prior2"]
            ),
        )

        ls_init = gp_options["ls_prior3"]
        self.covar_module.base_kernel.lengthscale = ls_init

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KExactGPModel(ExactGPModel):
    """
    Derived class that uses a KeOps-compatible Matern kernel for scalable computations.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood, kernel_type=KMaternKernel)


def build_and_optimize_gp(X: torch.Tensor, Y: torch.Tensor, use_keops: bool = False):
    """
    Modified EDBO+ function that returns trained GPyTorch GP on training inputs.
    """

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        GammaPrior(gp_options["noise_prior1"], gp_options["noise_prior2"])
    )

    likelihood.noise = gp_options["noise_prior3"]

    if use_keops:
        model = KExactGPModel(X, Y, likelihood).to(X)
    else:
        model = ExactGPModel(X, Y, likelihood).to(X)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(gp_options["noise_constraint"])
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
        ],
        lr=0.1,
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y.squeeze(-1).to(**tkwargs))

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood


def get_fitted_gpytorch_gp(
    train_X, train_Y, use_keops: bool = False
) -> Tuple[ModelListGP, List[Likelihood]]:
    """
    EDBO+ function for obtaining a trained BoTorch GP.
    From training inputs, return BoTorch compatible ModelList GP for acqfunc evaluation.
    """

    individual_models = []
    likelihoods = []

    for i in range(train_Y.shape[-1]):
        train_y_i = train_Y[..., i : i + 1]
        gp, likelihood = build_and_optimize_gp(train_X, train_y_i, use_keops=use_keops)

        model_i = SingleTaskGP(
            train_X=train_X,
            train_Y=train_y_i,
            covar_module=gp.covar_module,
            likelihood=likelihood,
        )

        model_i.eval()
        likelihood.eval()

        individual_models.append(model_i)
        likelihoods.append(likelihood)

    bigmodel = ModelListGP(*individual_models)

    return bigmodel, likelihoods


def get_fitted_botorch_gp(
    X: torch.Tensor,
    Y: torch.Tensor,
    use_ard: bool = True,
    fit_gpytorch_options: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelListGP, List[Likelihood]]:
    """
    Trains a Gaussian Process model with specified kernel and likelihood.
    Note that independent GPs are trained to model each output, and the kernel parameters are the same for each.

    Args:
    - X (torch.Tensor): Input features tensor, normalised.
    - Y (torch.Tensor): Output/target tensor, standardised.
    - use_ard (bool): Whether to use Automatic Relevance Determination.
        - when use_ard is set to True, the kernel is allowed to learn a separate length scale for each input feature,
          allowing the model to determine which features are more important in predicting the target variable
    - fit_gpytorch_options (Dict[str, Any], optional): Options for fitting the model.

    GP settings and code adapted from: https://arxiv.org/abs/2109.10964

    Returns:
    - ModelListGP object containing fitted models for both outputs
    - List of likelhoods for both models, which specifies how outputs are generated by GP latent functions.
    """

    models = []
    likelihoods = []

    for i in range(Y.shape[-1]):
        # Kernel setup with ARD and ScaleKernel
        ard_num_dims = X.shape[-1] if use_ard else 1

        kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=GammaPrior(
                    gp_options["ls_prior1"], gp_options["ls_prior2"]
                ),
            ),
            outputscale_prior=GammaPrior(
                gp_options["out_prior1"], gp_options["out_prior2"]
            ),
        )

        # Likelihood setup with noise constraints and prior
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(gp_options["noise_constraint"]),
            noise_prior=GammaPrior(
                gp_options["noise_prior1"], gp_options["noise_prior2"]
            ),
        )

        # GP model for each output dimension
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y[:, i : i + 1],
            covar_module=kernel,
            likelihood=likelihood,
        )

        model.train()
        likelihood.train()

        models.append(model)
        likelihoods.append(likelihood)

    model_list = ModelListGP(*models)

    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    fit_gpytorch_model(mll, options=fit_gpytorch_options)

    return model_list, likelihoods
