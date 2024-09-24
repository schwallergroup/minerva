# Execution scripts

This directory contains the scripts used to run optimisation rounds and obtain optimiser suggestions on the experimental ML HTE campaign described in the manuscript. The optimiser suggestions for each iteration can be found in the `experimental_campaigns/experiments/publication/ML_plate_suggestions` directory.

```run_BO_iteration.py --current_iteration <iteration>``` was used to run Bayesian optimisation iterations 2 to 5, obtaining ML plates 2 to 5a. This script requires 5-10 minutes to execute on our specified workstation.

`run_exploitative_iteration.py` contains the code used to run the exploitative iteration for ML plate 5b.