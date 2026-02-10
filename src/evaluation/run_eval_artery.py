import numpy as np
import pandas as pd
import os

from scipy.linalg import block_diag

from src.association.StochasticOptimization import T2TA_SO
from src.association.best_asso import gt_association
from src.association.greedy import greedy_multidim, greedy_sensorwise
from src.util_artery import compute_gospa_artery, fuse_estimates


def run_eval(
        root_path,
        run,
        rules,
        penetration_rate,
        num_sweeps,
        start_time=15.0,
        end_time=25.0,
        save_tracks_time=0.5,
        invalidate_time=0.5,
):
    path = os.path.join(root_path, str(run), rules, penetration_rate)
    # load and parse files
    tracks_recieved = pd.read_csv(os.path.join(path, "log_tracks.csv"), dtype={"external_id": "str"})
    gt_positions = pd.read_csv(
        os.path.join(path, "log_gt_positions.csv"),
        dtype={"external_id": "str"},
    )
    sensor_pos = (
        pd.read_csv(os.path.join(path, "log_sensor_pos.csv")).replace("RSU", -1).astype({"src_station": "int32"})
    )

    tracks_recieved["mean"] = tracks_recieved["mean"].apply(lambda x: np.array([float(xi) for xi in x.split(";")]))
    tracks_recieved["cov"] = tracks_recieved["cov"].apply(
        lambda x: np.array([[float(xj) for xj in xi.split(" ")] for xi in x.split(";")])
    )
    sensor_pos["position"] = sensor_pos["position"].apply(lambda x: np.array([float(xi) for xi in x.split(" ")]))

    # result lists
    fused_tracks = []
    best_singles = []
    associations_list = []
    tracklist = []

    times = tracks_recieved.query(f"(time>={start_time})&(time<{end_time})")["time"].unique()

    for time in times:

        if rules == 'etsi':
            track_group = tracks_recieved.query(f"(time<={time})&(time>{time - save_tracks_time})")
            # Filter out double tracks from same sensor
            idx = track_group.groupby(["src_station", "track_id"])["lastMeasurement"].idxmax()
            track_group = track_group.loc[idx]

        else:
            track_group = tracks_recieved.query(f"time=={time}")

        # filter tracks whose update has been too long ago
        track_group = track_group.query(f"{time} - lastMeasurement < {invalidate_time}").copy()

        print(time)
        n = len(track_group)

        # propagate tracks
        propagated = track_group.apply(
            lambda t: ukf_predict(t["mean"], t["cov"], time - t["lastMeasurement"]),
            axis=1,
            result_type="expand",
        )
        track_group.loc[:, ("mean", "cov")] = propagated

        # precompute inverse covariances and determinants
        track_group.loc[:, "cov_inv"] = track_group.apply(lambda x: np.linalg.inv(x["cov"]), axis=1)
        track_group.loc[:, "cov_det"] = track_group.apply(lambda x: np.linalg.det(x["cov"]), axis=1)

        # tracks as np array
        tracks = np.zeros((n, 6))
        tracks[:, :-1] = np.array(track_group["mean"].to_list())  # [:, :2]
        tracks[:, -1] = track_group["src_station"].to_numpy()

        # sensor positions
        sensor_pos_t = sensor_pos.query(f"(time<={time})&(time>{time - save_tracks_time})")
        # Filter out double positions from same sensor
        idx = sensor_pos_t.groupby("src_station")["time"].idxmax()
        sensor_pos_t = sensor_pos_t.loc[idx]

        sensor_pos_t = np.concatenate(
            (sensor_pos_t["position"].to_list(), sensor_pos_t["src_station"].to_numpy()[:, None]),
            axis=1,
        )
        sensor_pos_t = sensor_pos_t[sensor_pos_t[:, -1] >= 0]  # filter out RSU
        num_sensors = sensor_pos_t.shape[0]

        if not (set(tracks[:, -1]) <= set(sensor_pos_t[:, -1])):
            print("Error in sensor computation!")

        # set up sampling
        so_const_gl = T2TA_SO(pd_method="const", pd_est_factor=4, likelihood="gl")
        so_const_ml = T2TA_SO(pd_method="const", pd_est_factor=4, likelihood="ml")
        so_sensor_dist_gl = T2TA_SO(
            pd_method="sensor_dist",
            max_pd=0.97,
            min_pd=0.97,
            else_pd=0.15,
            max_dist=95,
            likelihood="gl",
        )
        so_sensor_dist_ml = T2TA_SO(
            pd_method="sensor_dist",
            max_pd=0.97,
            min_pd=0.97,
            else_pd=0.15,
            max_dist=95,
            likelihood="ml",
        )

        # associate
        so_asso_const_gl, _ = so_const_gl.associate(tracks, num_sweeps, num_sensors, tracks_df=track_group,
                                                    return_best=True)
        so_asso_const_ml, _ = so_const_ml.associate(tracks, num_sweeps, num_sensors, tracks_df=track_group,
                                                    return_best=True)
        so_asso_sensor_dist_gl, _ = so_sensor_dist_gl.associate(
            tracks, num_sweeps, num_sensors, sensor_pos=sensor_pos_t, return_best=True, tracks_df=track_group
        )
        so_asso_sensor_dist_ml, _ = so_sensor_dist_ml.associate(
            tracks, num_sweeps, num_sensors, sensor_pos=sensor_pos_t, return_best=True, tracks_df=track_group
        )

        best_asso = gt_association(track_group)
        greedy_asso_gl = greedy_multidim(tracks, merge=True, likelihood="gl", tracks_df=track_group)
        greedy_asso_ml = greedy_multidim(tracks, merge=True, likelihood="ml", tracks_df=track_group)
        greedy_asso_euclid = greedy_multidim(tracks, merge=True, likelihood="euclid", tracks_df=track_group)

        greedy_old_asso_gl = greedy_multidim(tracks, merge=False, likelihood="gl", tracks_df=track_group)
        greedy_old_asso_ml = greedy_multidim(tracks, merge=False, likelihood="ml", tracks_df=track_group)
        greedy_old_asso_euclid = greedy_multidim(tracks, merge=False, likelihood="euclid", tracks_df=track_group)

        greedy_2D_asso_gl = greedy_sensorwise(tracks, likelihood="gl", tracks_df=track_group)
        greedy_2D_asso_ml = greedy_sensorwise(tracks, likelihood="ml", tracks_df=track_group)
        greedy_2D_asso_euclid = greedy_sensorwise(tracks, likelihood="euclid", tracks_df=track_group)

        # save results
        associations = {
            "SO_const_gl": so_asso_const_gl,
            "SO_const_ml": so_asso_const_ml,
            "SO_sensor_dist_gl": so_asso_sensor_dist_gl,
            "SO_sensor_dist_ml": so_asso_sensor_dist_ml,
            "best": best_asso,
            "greedy_gl": greedy_asso_gl,
            "greedy_ml": greedy_asso_ml,
            "greedy_euclid": greedy_asso_euclid,
            "greedy_no_merge_gl": greedy_old_asso_gl,
            "greedy_no_merge_ml": greedy_old_asso_ml,
            "greedy_no_merge_euclid": greedy_old_asso_euclid,
            "greedy_2D_gl": greedy_2D_asso_gl,
            "greedy_2D_ml": greedy_2D_asso_ml,
            "greedy_2D_euclid": greedy_2D_asso_euclid,
        }
        for method in associations:
            track_group.loc[track_group.index, method] = associations[method]

        track_group.loc[track_group.index, "time"] = time
        tracklist.append(track_group)

        # fuse results
        fused_tracks += fuse_estimates(time, track_group, associations)
        associations["time"] = time
        associations_list.append(associations)

        best_single = (
            track_group.query(f"src_station=={track_group['src_station'].value_counts().idxmax()}")
            .drop(columns=["external_id", "src_station", "track_id", "lastMeasurement"])
            .rename(columns={"mean": "fused_estimate", "cov": "fused_cov"})
        )
        state = np.array(best_single["fused_estimate"].to_list())
        best_single["pos_x"] = state[:, 0]
        best_single["pos_y"] = state[:, 1]
        best_single["method"] = "best_single"
        best_singles.append(best_single)

    fused_tracks = pd.concat([pd.DataFrame(fused_tracks), pd.concat(best_singles)])

    # save fused tracks
    fused_tracks.to_pickle(
        os.path.join(root_path, str(run), f"fused_tracks_{run}_{rules}_{penetration_rate}.pck")
    )

    # save associations
    associations_list = pd.DataFrame(associations_list).set_index("time")
    associations_list.to_pickle(
        os.path.join(root_path, str(run), f"associations_{run}_{rules}_{penetration_rate}.pck")
    )

    # compute gospa
    errors = compute_gospa_artery(fused_tracks, gt_positions)
    if errors.empty:
        return

    gospa_avg = (
        errors.groupby("method")
        .agg(
            avg_gospa=("gospa_distance", "mean"),
            avg_gospa_localization=("gospa_localization", "mean"),
            avg_gospa_missed=("gospa_missed", "mean"),
            avg_gospa_false=("gospa_false", "mean"),
            avg_gospa_po=("gospa_distance_po", "mean"),
            avg_gospa_localization_po=("gospa_localization_po", "mean"),
            avg_gospa_missed_po=("gospa_missed_po", "mean"),
            avg_gospa_false_po=("gospa_false_po", "mean"),
        )
        .reset_index()
    )
    gospa_avg["run"] = run
    gospa_avg["rules"] = rules
    gospa_avg["penetration_rate"] = penetration_rate

    # save averaged gospa
    gospa_avg.to_pickle(
        os.path.join(root_path, str(run), f"gospa_agg_{run}_{rules}_{penetration_rate}.pck")
    )
    # save gospa per time step
    errors.to_pickle(os.path.join(root_path, str(run), f"gospa_{run}_{rules}_{penetration_rate}.pck"))

    # save tracks with associations
    tracklist = pd.concat(tracklist)
    tracklist.to_pickle(
        os.path.join(root_path, str(run), f"tracks_{run}_{rules}_{penetration_rate}.pck")
    )


def circular_mean(angles, weights):
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    mean_angle = np.arctan2(s, c)
    return mean_angle


def normal_avg(states, weights):
    if type(weights) is float:
        return np.sum(states, axis=0) * weights

    return np.sum(weights[:, None] * states, axis=0)


def ct_motion(x_old, T, process_noise=np.array([0, 0])):
    x1, x2, v1, v2, yaw = x_old
    if np.abs(yaw) < 1e-7:
        x1_new = x1 + v1 * T
        x2_new = x2 + v2 * T
    else:
        x1_new = x1 + (v1 / yaw) * np.sin(yaw * T) - (v2 / yaw) * (1 - np.cos(yaw * T))
        x2_new = x2 + (v1 / yaw) * (1 - np.cos(yaw * T)) + (v2 / yaw) * np.sin(yaw * T)

    v1_new = v1 * np.cos(yaw * T) - v2 * np.sin(yaw * T)
    v2_new = v1 * np.sin(yaw * T) + v2 * np.cos(yaw * T)

    G = np.array([[T ** 2 / 2.0, 0, 0], [0, T ** 2 / 2.0, 0], [T, 0, 0], [0, T, 0], [0, 0, T]])

    x_new = np.array([x1_new, x2_new, v1_new, v2_new, yaw]) + G @ process_noise

    return x_new


def ukf_predict(x_old, cov_old, T):
    mean_ext = np.concatenate((x_old, np.zeros(3)))
    cov_ext = block_diag(cov_old, np.diag([5 ** 2, 5 ** 2, (0.08 * np.pi) ** 2]))

    A = np.linalg.cholesky(cov_ext)
    L = 8

    weights = np.zeros(2 * L + 1)
    W0 = 1 / L
    weights[0] = W0
    weights[1:] = (1 - W0) / (2 * L)

    samples = np.zeros((2 * L + 1, 8))
    samples[0] = mean_ext.copy()
    for j in range(1, L + 1):
        samples[j] = mean_ext + (L / (1 - W0)) ** 0.5 * A[:, j - 1]
        samples[L + j] = mean_ext - (L / (1 - W0)) ** 0.5 * A[:, j - 1]

    predicted_samples = np.zeros((2 * L + 1, 5))
    for i in range(2 * L + 1):
        predicted_samples[i] = ct_motion(samples[i, :5], T, samples[i, 5:])

    predicted_state = np.average(predicted_samples, weights=weights, axis=0)

    pred_diff = predicted_samples - predicted_state

    predicted_cov = np.sum(
        weights[:, None, None] * pred_diff[:, :, None] @ pred_diff[:, None, :],
        axis=0,
    )

    return {"mean": predicted_state, "cov": predicted_cov}
