"""
Run iteration of Bayesian optimisation from training data.

This script was used to perform Bayesian optimisation on experimental data generated in
the HTE reaction optimisation campaign described in the manuscript associated with this
code repository.

It loads training data depending on the current iteration, processes it,
and generates new experiments based on the optimization results.
"""

import logging
import argparse
from typing import Dict, Any, Tuple

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from minerva.utils import get_index_from_lookup
from minerva.bayesopt import BayesianOptimisation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TRAIN_DATA_PATH = "../experimental_campaigns/experiments/publication/ML_training_data/ML_{}_train_data.csv"
CHEM_SPACE_PATH = "../experimental_campaigns/design_spaces/ni_suzuki_chemical_space.csv"
DESCRIPTOR_INDEX = (
    7  # our reaction condition representation starts from column 7 in chem_space
)
OBJECTIVE_COLUMNS = ["P_A%", "Selectivity_A%"]


def load_training_data(current_iteration: int) -> pd.DataFrame:
    """
    Load training data based on the current iteration.

    For example, if we are currently obtaining ML suggestions for the fourth iteration,
    we would use training data from iterations/plates 1, 2, and 3, and the train data path
    would be:

    ../experimental_campaigns/experiments/publication/ML_training_data/ML_plate_123_train_data.csv

    Note that ML optimisation can only be performed starting from iteration 2 with data obtained
    from Sobol initialisation of iteration/plate 1.
    
    We obtained experimental data up to iteration 5 in the HTE campaign.
    """
    plate_numbers = "".join(str(i) for i in range(1, current_iteration))
    train_plates = f"plate_{plate_numbers}"
    train_data_path = TRAIN_DATA_PATH.format(train_plates)

    logging.info(f"Current iteration: {current_iteration}")
    logging.info(f"Using training data from: {train_data_path}")

    return pd.read_csv(train_data_path, index_col=0)


def load_chemical_space() -> pd.DataFrame:
    """Load the pre-defined chemical space."""
    return pd.read_csv(CHEM_SPACE_PATH, index_col=0)


def preprocess_data(
    plate_data: pd.DataFrame, chem_space: pd.DataFrame, tkwargs: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess the data for Bayesian Optimization.

    We first obtain the featurised chemical space representation, and convert it into a torch tensor.
    We then obtain normalised train_x and standardised train_y_std for training the GP.

    Returns (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    - X_main (torch.Tensor): Featurised search space to evaluate ML model over
    - train_x (torch.Tensor): Normalised training inputs of already conducted experiments
    - train_y_std (torch.Tensor): Standardised training outputs of already conducted experiments
    """
    # we obtain the featurised chemical space representation from the chemical space dataframe
    chem_descriptors = chem_space.iloc[:, DESCRIPTOR_INDEX:]

    # we convert the featurised chemical space into a tensor
    X_main = torch.tensor(chem_descriptors.to_numpy()).to(**tkwargs)

    # get train_x, which is already normalised
    plate_data.drop(labels=["rxn_id"], axis=1, inplace=True)
    columns_regression = plate_data.columns.drop(OBJECTIVE_COLUMNS).to_list()
    train_x = torch.tensor(plate_data[columns_regression].to_numpy()).to(**tkwargs)

    # standardise train_y to train_y_std
    scaler_objectives = StandardScaler()
    train_y = plate_data[OBJECTIVE_COLUMNS].to_numpy()
    train_y_std = scaler_objectives.fit_transform(train_y)
    train_y_std = torch.tensor(train_y_std).to(**tkwargs)

    return X_main, train_x, train_y_std


def main(args):
    """Run BO with experimental data"""
    tkwargs = {
        "dtype": getattr(torch, args.dtype),
        "device": torch.device(args.device),
    }

    # import training data based on current iteration of optimisation (up to iteration 5)
    current_iteration = args.current_iteration

    # load plate training_data
    plate_data = load_training_data(current_iteration)

    # get already conducted reactions
    conducted_experiments = plate_data['rxn_id']

    # load chemical reaction search space and remove already conducted experiments
    chem_space = load_chemical_space()
    chem_space = chem_space[~chem_space['rxn_id'].isin(conducted_experiments)]

    X_main, train_x, train_y_std = preprocess_data(plate_data, chem_space, tkwargs)

    Campaign = BayesianOptimisation(device=tkwargs, seed=args.seed)

    new_x = Campaign.run_optimisation_iteration(
        train_x=train_x,
        train_y_std=train_y_std,
        x_space=X_main,
        batch_size=args.batch_size,
        kernel=args.kernel,
        constrain_strategy="nested",
        n_unique_features=args.n_unique_features,
        feature_index=args.feature_index,
        acqf_samples=args.acqf_samples,
        max_batch_eval=1024,  # reduce for large train_data
    )

    # looks up new_x tensor in chemical space to get new experiments
    new_index = get_index_from_lookup(new_x, X_main)
    plate_df = chem_space.iloc[new_index]
    plate_df = plate_df.iloc[:, :DESCRIPTOR_INDEX]
    print(plate_df)

    if current_iteration == 5:
        plate_df.to_csv("ML_plate_5a_suggestions.csv")
    else:
        plate_df.to_csv(f"ML_plate_{current_iteration}_suggestions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization")
    parser.add_argument("--seed", type=int, default=49, help="Seed for reproducibility")
    parser.add_argument(
        "--dtype", type=str, default="double", help="Data type for PyTorch tensors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computations",
    )

    # optimisation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size for Bayesian Optimization",
    )
    parser.add_argument(
        "--n_unique_features",
        type=int,
        default=2,
        help="Number of allowed unique values of constrained features in a batch",
    )
    parser.add_argument(
        "--feature_index",
        type=int,
        default=-1,
        help="index of feature in X to be constrained, e.g. temperature",
    )
    parser.add_argument(
        "--acqf_samples",
        type=int,
        default=1024,
        help="Number of samples for acquisition function, reduce if running as test",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="edboplus",
        help="Kernel hyperparameters for Gaussian Process",
    )

    parser.add_argument(
        "--current_iteration",
        type=int,
        default=None,
        help="Current iteration to obtain ML suggestions for, up to iteration 5.",
        choices=[2, 3, 4, 5]
    )

    args = parser.parse_args()
    main(args)
