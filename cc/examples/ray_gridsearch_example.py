from typing import Any

import numpy as np
import ray
from ray import air, tune

from cc.core.config import disable_compile_warn, disable_tqdm, force_cpu_backend


def YOUR_FUNCTION(config: dict[str, Any]) -> float:
    # do some work
    score = config["x"] ** 2 + config["y"] * 0.7 - config["z"]
    return -float(score)


def objective(config):
    force_cpu_backend()
    # Otherwise the stdout will be messy
    disable_tqdm()
    disable_compile_warn()

    return YOUR_FUNCTION(config)


ray.init()
# If you are working on the NHR FAU Cluster, and want to scale
# your gridsearch across multiple nodes using the `slurm-template.sh`
# Then you need to call
# >> import os
# >> ray.init(address=os.environ.get("ip_head", None))
# An alternative that also works
# >> ray.init(address="auto", _redis_password = os.environ["redis_password"])
# from https://discuss.ray.io/t/ray-on-slurm-unmatched-raylet-address/8509/3

search_space = {
    "x": tune.grid_search([0, 1, 2]),
    "y": tune.grid_search(np.arange(2)),
    "z": tune.grid_search([17.2]),
}

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 2}),  # <- two cpus per job
    tune_config=tune.TuneConfig(
        num_samples=1,  # <- 1 sample *per* grid point
        time_budget_s=24 * 3600,  # maximum runtime in seconds
        mode="min",  # either `min` or `max`
    ),
    run_config=air.RunConfig(
        log_to_file=("my_stdout.log", "my_stderr.log"), local_dir="ray_results"
    ),
    param_space=search_space,
)

tuner.fit()
ray.shutdown()

# This will create a folder, let's call it `objective_dir` in `ray_results`
# and store all logs in there.
# To later transfer the gridsearch results to your local machine you could e.g. do
# >> tune.ExperimentAnalysis(`objective_dir`).dataframe().to_pickle("./results.pkl")
# Then transfer the `results.pkl` to your local machine and use
# >> pd.read_pickle("results.pkl")

# ----------------
# Q: How to handle trials that error due to memory pressure?
# A: The easiest way is to simply restore the gridsearch sweep but only re-run
#    all trials that errored (or that haven't run yet)
#    This can be done with
#    >> tune.Tuner.restore(`path_to_objective_dir`, restart_errored=True).fit()
