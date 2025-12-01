import sys

sys.path.append(".")
from src.evaluation.run_eval import run_t2ta_eval

num_mc_runs = 10

detection_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
sigmas = [1.0, 2.0]
folder = f"results/minimal"
for sigma in sigmas:
    for detection_prob in detection_probs:
        run_t2ta_eval(
            num_objects=8,
            num_sensors=5,
            num_mc_scenarios=num_mc_runs,
            num_sweeps=100,
            detection_prob=detection_prob,
            base_folder=folder,
            scenario="random_close",
            spatial_sig=sigma,
            convergence=False,
            return_best_num=5,
            do_orig_greedy=True,
            do_sd_assign=False,
            likelihood_variants=['ml_const', 'gl_const', 'euclid']
        )

detection_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
sigmas = [1.0, 2.0]
folder = f"results/big"
for sigma in sigmas:
    for detection_prob in detection_probs:
        run_t2ta_eval(
            num_objects=20,
            num_sensors=12,
            num_mc_scenarios=num_mc_runs,
            num_sweeps=200,
            detection_prob=detection_prob,
            base_folder=folder,
            scenario="random",
            spatial_sig=sigma,
            convergence=False,
            return_best_num=10,
            do_orig_greedy=True,
            do_sd_assign=False,
            likelihood_variants=['gl_const', 'ml_const', 'euclid']
        )

# convergence
detection_prob = 0.8
sigma = 2.0

folder = f"results/convergence"
run_t2ta_eval(
    num_objects=20,
    num_sensors=12,
    num_mc_scenarios=num_mc_runs,
    num_sweeps=300,
    detection_prob=detection_prob,
    base_folder=folder,
    scenario="random",
    spatial_sig=sigma,
    convergence=True,
    do_orig_greedy=True,
    do_sd_assign=False,
    likelihood_variants=['ml_const']
)
