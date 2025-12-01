import datetime

import pandas as pd
import os
from src.abspath import root_folder
from src.association.Likelihood import Likelihood
from src.utils import compute_gospa, sort_gospa
import numpy as np
from scipy.stats import multivariate_normal
import random

from src.association.greedy import greedy_multidim, greedy_sensorwise
from src.association.StochasticOptimization import T2TA_SO


def run_t2ta_eval(
        num_objects,
        num_sensors,
        num_mc_scenarios,
        num_sweeps,
        detection_prob,
        base_folder,
        scenario="random",
        spatial_sig=2,
        convergence=False,
        likelihood_variants=["ml_const"],
        return_best_num=1,
        do_orig_greedy=False,
        do_sd_assign=False,
):
    if do_sd_assign:
        from src.association.SDAssign import SDAssign

        sd_optimizer = SDAssign()

    folder = os.path.join(
        root_folder,
        base_folder,
        scenario,
        f"{num_objects}objects_{num_sensors}sensors/{num_sweeps}samples/sig={spatial_sig}/pD={detection_prob}",
    )
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    rnd_run_id = random.randint(0, 999999999999)

    if num_mc_scenarios < 10:
        save_every = 1
    else:
        save_every = 10

    results_df = []
    objects_df = []
    for msc in range(num_mc_scenarios):

        print("mc run scenario : ", msc)
        results = []

        # generate ground truth object positions
        if scenario == "random":
            gt = random_objects(num_objects, 50)
        elif scenario == "random_close":
            gt = random_objects(num_objects, 30)
        elif scenario == "random_big":
            gt = random_objects(num_objects, 100)

        # sample tracks
        generate_tracks = True
        while generate_tracks:
            tracks = []
            perfect_asso = []
            for s in range(num_sensors):
                if convergence:  # same number of tracks in every run
                    num_tracks = int(np.rint(detection_prob * num_objects))  # rounding
                    idx = np.arange(gt.shape[0])
                    idx_rnd = np.random.choice(idx, num_tracks, replace=False)
                    idx_rnd.sort()
                    for i in idx_rnd:
                        tracks.append(
                            np.concatenate(
                                (gt[i] + multivariate_normal.rvs(mean=np.zeros(2), cov=np.eye(2) * spatial_sig ** 2),
                                 [s])
                            )
                        )
                        perfect_asso.append(i)

                else:
                    for i, obj in enumerate(gt):
                        if np.random.random() < detection_prob:
                            tracks.append(
                                np.concatenate(
                                    (obj + multivariate_normal.rvs(mean=np.zeros(2),
                                                                   cov=np.eye(2) * spatial_sig ** 2), [s])
                                )
                            )
                            perfect_asso.append(i)

            if not tracks:
                continue
            tracks = np.array(tracks)

            if len(set(tracks[:, -1])) > 2:
                # only want at least 3D problem
                generate_tracks = False
            else:
                print(f"only {len(set(tracks[:, -1]))} sensors")

        # ground truth association
        perfect_asso = np.array(perfect_asso) + 1

        # shuffle tracks
        shuffle_idx = np.arange(tracks.shape[0])
        np.random.shuffle(shuffle_idx)
        tracks = tracks[shuffle_idx]
        perfect_asso = perfect_asso[shuffle_idx]

        for likelihood in likelihood_variants:
            # associate
            if likelihood == 'first': continue
            SO_sampler = T2TA_SO(
                pd_method="const",
                spatial_sig=spatial_sig,
                likelihood=likelihood,
            )

            if convergence:
                (so_asso, so_weight, so_samples, so_actions) = SO_sampler.associate(
                    tracks, num_sweeps, num_sensors, p_D=detection_prob, return_best=False
                )
                gospa_batch = [compute_gospa(gt, tracks, s)["distance"] for s in so_asso]
                gospa_batch = np.array(gospa_batch)
                best_idx = so_weight.argmax()
                results.append(
                    {
                        "method": f"SO_{likelihood}",
                        "associations": so_asso,
                        "best_asso": so_asso[best_idx],
                        "weight": so_weight[best_idx],
                        "weights": so_weight,
                        "gospa": gospa_batch[best_idx],
                        "gospa_batch": gospa_batch,
                        "saved_samples": so_samples,
                        "saved_actions": so_actions,
                        "best_idx": best_idx,
                    }
                )

            else:
                so_asso, so_weight = SO_sampler.associate(
                    tracks, num_sweeps, num_sensors, p_D=detection_prob, return_best=True,
                    return_best_num=return_best_num
                )
                if return_best_num > 1:
                    gospa_batch = [compute_gospa(gt, tracks, s)["distance_po"] for s in so_asso]

                    gospa_batch = np.array(gospa_batch)
                    best_idx = so_weight.argmax()
                    results.append(
                        {
                            "method": f"SO_{likelihood}",
                            "associations": so_asso,
                            "best_asso": so_asso[best_idx],
                            "weight": so_weight[best_idx],
                            "weights": so_weight,
                            "gospa_batch": gospa_batch,
                            "best_idx": best_idx,
                            **compute_gospa(gt, tracks, so_asso[best_idx]),
                        }
                    )
                else:
                    results.append(
                        {
                            "method": f"SO_{likelihood}",
                            "best_asso": so_asso,
                            "weight": so_weight,
                            "best_idx": 0,
                            **compute_gospa(gt, tracks, so_asso),
                        }
                    )

            lik = Likelihood('const', spatial_sig=spatial_sig, likelihood=likelihood)
            lik.set_tracks(tracks, num_sensors, p_D=detection_prob)
            greedy_asso = greedy_multidim(tracks, likelihood=likelihood, spatial_sig=spatial_sig)

            results.append(
                {
                    "method": f"greedy_{likelihood}",
                    "weight": lik.compute_weights(greedy_asso)[0],
                    **compute_gospa(gt, tracks, greedy_asso),
                    "best_idx": 0,
                    "best_asso": greedy_asso,
                }
            )
            if do_orig_greedy:
                greedy_old_asso = greedy_multidim(
                    tracks, merge=False, likelihood=likelihood, spatial_sig=spatial_sig
                )
                results.append(
                    {
                        "method": f"greedy_no_merge_{likelihood}",
                        "weight": lik.compute_weights(greedy_old_asso)[0],
                        **compute_gospa(gt, tracks, greedy_old_asso),
                        "best_idx": 0,
                        "best_asso": greedy_old_asso,
                    }
                )

            greedy_2D_asso = greedy_sensorwise(tracks, likelihood=likelihood, spatial_sig=spatial_sig)

            results.append(
                {
                    "method": f"greedy_2D_{likelihood}",
                    "weight": lik.compute_weights(greedy_2D_asso)[0],
                    **compute_gospa(gt, tracks, greedy_2D_asso),
                    "best_idx": 0,
                    "best_asso": greedy_2D_asso,
                }
            )

            if do_sd_assign and likelihood != 'euclid':
                sd_opt_asso = sd_optimizer.associate_sd(tracks, detection_prob, spatial_sig, likelihood)

                results.append(
                    {
                        "method": f"assign_SD_{likelihood}",
                        "weight": lik.compute_weights(sd_opt_asso)[0],
                        **compute_gospa(gt, tracks, sd_opt_asso),
                        "best_idx": 0,
                        "best_asso": sd_opt_asso,
                    }
                )

            results.append(
                {
                    "method": f"perfect_{likelihood}",
                    "weight": lik.compute_weights(perfect_asso)[0],
                    **compute_gospa(gt, tracks, perfect_asso),
                    "best_idx": 0,
                    "best_asso": perfect_asso,
                }
            )
        results = pd.DataFrame(results)
        results["mc_run_scenario"] = msc
        results_df.append(results)
        objects = pd.concat(
            [
                pd.DataFrame({"pos_x": gt[:, 0], "pos_y": gt[:, 1], "sensor": -1}),
                pd.DataFrame({"pos_x": tracks[:, 0], "pos_y": tracks[:, 1], "sensor": tracks[:, 2]}),
            ]
        )
        objects["mc_run_scenario"] = msc
        objects_df.append(objects)

        if (msc + 1) % save_every == 0:
            objects_df = pd.concat(objects_df)
            results_df = pd.concat(results_df)
            objects_df["rnd_id"] = rnd_run_id
            results_df["rnd_id"] = rnd_run_id
            if convergence:
                results_df["gospa_sort"] = results.apply(lambda x: sort_gospa(x, num_sweeps), axis=1)
                results_df = results_df.drop(columns=["associations", "weights", "gospa_batch", "saved_samples"])
            time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")
            # if not convergence:
            objects_df.to_pickle(os.path.join(folder, f"objects_{time}_{rnd_run_id}.pck"))
            results_df.to_pickle(os.path.join(folder, f"t2ta_{time}_{rnd_run_id}.pck"))
            print("saved: ", os.path.join(folder, f"objects_{time}_{rnd_run_id}.pck"))

            results_df = []
            objects_df = []


def random_objects(num_objects, scale):
    gt = np.random.random((num_objects, 2)) * scale
    return gt
