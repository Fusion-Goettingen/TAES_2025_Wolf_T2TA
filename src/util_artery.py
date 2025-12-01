import sys

import numpy as np
import pandas as pd

from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.types.state import State, StateVector

sys.path.append(".")


def compute_gospa_artery(fused, gt, c=10, p=1):
    # convert positions to states

    fused["State"] = fused[["pos_x", "pos_y"]].apply(
        lambda x: State(state_vector=StateVector([x.pos_x, x.pos_y])), axis=1
    )

    gt["State"] = gt[["pos_x", "pos_y"]].apply(
        lambda x: State(state_vector=StateVector([x.pos_x, x.pos_y])), axis=1
    )

    gospa = GOSPAMetric(c=c, p=p)

    def do_gospa(x):
        return gospa.compute_gospa_metric(x.fused_states, x.gt_states)

    # group states by method and time step, aggregate per method and timestep -> compute and gospa
    fused.set_index("time")
    fused_agg = fused.groupby(["time", "method"], group_keys=True)["State"].agg(
        fused_states=lambda x: list(x)
    )
    fused_agg = fused_agg.reset_index("method")
    gt_agg = gt.groupby("time")["State"].agg(gt_states=lambda x: list(x))

    # merge fused and gt state lists on timestep
    res = pd.merge(fused_agg, gt_agg, how="inner", on="time")

    # compute gospa
    if not res.empty:
        res["gospa"] = res.apply(lambda x: do_gospa(x), axis=1)
        res["num_objects"] = res.apply(lambda x: len(x.gt_states), axis=1)

        # unpack gospa values
        res["gospa_distance"] = res["gospa"].apply(lambda x: x[0].value["distance"])
        res["gospa_localization"] = res["gospa"].apply(
            lambda x: x[0].value["localisation"]
        )
        res["gospa_missed"] = res["gospa"].apply(lambda x: x[0].value["missed"])
        res["gospa_false"] = res["gospa"].apply(lambda x: x[0].value["false"])

        for metric in ["distance", "localization", "missed", "false"]:
            res[f"gospa_{metric}_po"] = res[f"gospa_{metric}"] / res["num_objects"]

    return res


def fuse_estimates(time, tracks, associations):
    results = []
    for method in associations:
        for id, cluster in tracks.groupby(method):
            est, cov = ifci(cluster)
            results.append(
                {
                    "time": time,
                    "method": method,
                    "fused_estimate": est,
                    "fused_cov": cov,
                    "pos_x": est[0],
                    "pos_y": est[1],
                }
            )
        aaah = 42

    return results


def ifci(tracks, only_position=True):
    n = len(tracks)
    means = np.array(tracks["mean"].tolist())
    covs = np.array(tracks["cov"].tolist())
    if n == 1:
        return means[0], covs
    if only_position:
        means = means[:, :2]
        covs = covs[:, :2, :2]

    if n == 1:
        return means.squeeze(), covs.squeeze()

    information = np.linalg.inv(covs)
    inf_sum = np.sum(information, axis=0)
    inf_sum_det = np.linalg.det(inf_sum)
    inf_det = np.linalg.det(information)
    inf_diff_det = np.linalg.det(inf_sum - information)

    weights = (inf_sum_det - inf_diff_det + inf_det) / (n * inf_sum_det + np.sum(inf_det - inf_diff_det))

    fused_inf = np.sum(weights[:, None, None] * information, axis=0)
    fused_cov = np.linalg.inv(fused_inf)

    fused_state = fused_cov @ np.sum(weights[:, None, None] * information @ means[:, :, None], axis=0)

    return fused_state.squeeze(), fused_cov
