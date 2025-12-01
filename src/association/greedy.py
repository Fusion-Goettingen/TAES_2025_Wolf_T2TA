import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from src.association.Likelihood import Likelihood, gaussian_likelihood


def greedy_multidim(
        tracks,
        merge=True,
        likelihood="euclid",
        spatial_sig=None,
        tracks_df=None,
):
    """
    Greedy (Merge) and Greedy (no Merge)
    :param tracks: np array of tracks with sensors in the last column
    :param merge: do merging
    :param likelihood: which likelihood to use
    :param spatial_sig: noise parameter for mc simulations
    :param tracks_df: dataframe with tracks for artery simulation
    :return: joint association
    """
    num_sensors = len(set(tracks[:, -1]))
    num_tracks = tracks.shape[0]

    # if there is only one sensor, all tracks are singletons
    if num_sensors < 2:
        return np.arange(num_tracks) + 1

    max_dist = 9999

    # compute cost matrix
    if likelihood == "euclid":  # euclidean distance
        dist = cdist(tracks[:, :2], tracks[:, :2])
        distance_th = 10
    elif likelihood == "ml_const":  # proposed likelihood constant covariance
        cov = (
                np.eye(tracks.shape[1] - 1) * spatial_sig ** 2 * 1.5
        )  # * 1.5 accounts for mean uncertainty in cluster with 2 tracks
        dist = np.zeros((num_tracks, num_tracks))
        for i in range(num_tracks):
            tmp = gaussian_likelihood(tracks[i, :-1], np.linalg.inv(cov), np.linalg.det(cov), tracks[i:, :-1])
            dist[i, i:] = tmp
            dist[i:, i] = tmp
        dist = -np.log(dist + 1e-300)
        distance_th = 15
    elif likelihood == "ml":  # proposed likelihood arbitrary covariances
        lik = Likelihood(pd_method="const", likelihood="ml")
        lik.set_tracks(tracks, num_sensors, tracks_df, p_D=0.5)
        dist = np.zeros((num_tracks, num_tracks))

        for i in range(num_tracks):
            for j in range(i, num_tracks):
                tmp = lik.cluster_lik({i, j})  # log space
                dist[i, j] = dist[j, i] = tmp
        dist -= np.log(0.5) * num_sensors  # "divide" by likelihood for cluster size
        dist = -dist  # neg log space
        distance_th = 20

    elif likelihood == "gl_const":  # generalized likelihood for constant covariance
        cov = np.eye(tracks.shape[1] - 1) * spatial_sig ** 2
        dist = np.zeros((num_tracks, num_tracks))
        for i in range(num_tracks):
            tmp = gaussian_likelihood(tracks[i, :-1], np.linalg.inv(cov), np.linalg.det(cov), tracks[i:, :-1])
            dist[i, i:] = tmp
            dist[i:, i] = tmp
        dist = -np.log(dist + 1e-300)
        distance_th = 15
    elif likelihood == "gl":  # generalized likelihood for constant covariance
        lik = Likelihood(pd_method="const", likelihood="gl")
        lik.set_tracks(tracks, num_sensors, tracks_df, p_D=0.5)
        dist = np.zeros((num_tracks, num_tracks))

        for i in range(num_tracks):
            for j in range(i, num_tracks):
                tmp = lik.cluster_lik({i, j})  # log space
                dist[i, j] = dist[j, i] = tmp
        dist -= np.log(0.5) * num_sensors  # "divide" by likelihood for cluster size
        dist = -dist  # neg log space
        distance_th = 20

    # exlude upper triangular matrix and pairs over distance threshold
    dist[np.triu_indices(num_tracks)] = max_dist
    dist[dist > distance_th] = max_dist

    # exclude pairs from same sensor
    for s in set(tracks[:, -1]):
        idx = tracks[:, -1] == s  # filter same sensor
        mask = idx[:, None] @ idx[None, :]  # lik of two tracks with same sensor is 0 (except to self)
        dist[mask] = max_dist
    rows, cols = np.unravel_index(np.argsort(dist.flatten()), shape=dist.shape)

    # initialize as all singletons
    joint_asso = np.zeros(num_tracks, dtype=np.int64)

    next_cluster = 1
    for r, c in zip(rows, cols):
        idx_r = joint_asso == joint_asso[r]
        idx_c = joint_asso == joint_asso[c]
        if dist[r, c] >= max_dist:
            continue

        # merge tracks if both are singletons
        if joint_asso[r] == 0 and joint_asso[c] == 0:
            joint_asso[r] = joint_asso[c] = next_cluster
            next_cluster += 1
        # if one track is singleton, move to cluster of other track, if sensor constraint is not violated
        elif joint_asso[r] == 0:
            if not tracks[r, -1] in tracks[idx_c, -1]:
                joint_asso[r] = joint_asso[c]
        elif joint_asso[c] == 0:
            if not tracks[c, -1] in tracks[idx_r, -1]:
                joint_asso[c] = joint_asso[r]
        else:  # both in cluster
            # merge clusters if sensor requirement is not violated
            if merge and joint_asso[c] != joint_asso[r]:
                if set(tracks[idx_c, -1]).isdisjoint(set(tracks[idx_r, -1])):  # all sensors different
                    joint_asso[idx_c] = joint_asso[r]  # merge clusters

        # exclude all pairs conflicting with current pair
        tr = tracks[r, -1]
        tc = tracks[c, -1]
        dist[r, tracks[:, -1] == tc] = max_dist
        dist[tracks[:, -1] == tr, c] = max_dist

    # singletons as separate clusters
    singleton_idx = joint_asso == 0
    joint_asso[singleton_idx] = np.arange(singleton_idx.sum()) + joint_asso.max() + 1
    return joint_asso


def greedy_sensorwise(tracks, likelihood="euclid", spatial_sig=None, tracks_df=None):
    """
    Sensorwise optimization
    :param tracks: np array of tracks with sensors in the last column
    :param likelihood: which likelihood to use
    :param spatial_sig: noise parameter for mc simulations
    :param tracks_df: dataframe with tracks for artery simulation
    :return: joint association
    """
    num_tracks = tracks.shape[0]
    num_sensors = len(set(tracks[:, -1]))

    sensors = list(set(tracks[:, -1]))
    clusters = {}
    joint_asso = np.zeros(num_tracks)

    # put tracks of first sensor in different clusters
    idx = np.where(tracks[:, -1] == sensors[0])[0]
    for i, t in enumerate(idx):
        clusters[i + 1] = [t]
        joint_asso[t] = i + 1

    num_clusters = idx.shape[0]

    for i in range(1, len(sensors)):
        # tracks from next sensor
        new_tracks_idx = np.where(tracks[:, -1] == sensors[i])[0]
        new_tracks = tracks[new_tracks_idx, :-1]

        # tracks that have been added last
        old_tracks_idx = [clusters[c][-1] for c in clusters]
        old_tracks = tracks[old_tracks_idx, :-1]

        # compute distance between old and new tracks
        if likelihood == "euclid":
            dist = cdist(new_tracks, old_tracks)
            distance_th = 10
        elif likelihood == "ml_const":
            # * 1.5 accounts for mean uncertainty in cluster with 2 tracks
            cov = np.eye(tracks.shape[1] - 1) * spatial_sig ** 2 * 1.5
            dist = np.zeros((new_tracks.shape[0], old_tracks.shape[0]))
            for n, new in enumerate(new_tracks):
                tmp = gaussian_likelihood(new, np.linalg.inv(cov), np.linalg.det(cov), old_tracks)
                dist[n, :] = tmp
            dist = -np.log(dist + 1e-16)
            distance_th = 15
        elif likelihood == "ml":
            lik = Likelihood("const", likelihood="ml")
            lik.set_tracks(tracks, num_sensors, tracks_df, p_D=0.5)
            dist = np.zeros((new_tracks.shape[0], old_tracks.shape[0]))

            for i in range(new_tracks.shape[0]):
                for j in range(old_tracks.shape[0]):
                    tmp = lik.cluster_lik({new_tracks_idx[i], old_tracks_idx[j]})  # log space
                    dist[i, j] = tmp
            dist -= np.log(0.5) * num_sensors  # "divide" by likelihood for cluster size
            dist = -dist  # neg log space
            distance_th = 20

        elif likelihood == "gl_const":
            cov = np.eye(tracks.shape[1] - 1) * spatial_sig ** 2
            dist = np.zeros((new_tracks.shape[0], old_tracks.shape[0]))
            for n, new in enumerate(new_tracks):
                tmp = gaussian_likelihood(new, np.linalg.inv(cov), np.linalg.det(cov), old_tracks)
                dist[n, :] = tmp
            dist = -np.log(dist + 1e-16)
            distance_th = 15
        elif likelihood == "gl":
            lik = Likelihood("const", likelihood="gl")
            lik.set_tracks(tracks, num_sensors, tracks_df, p_D=0.5)
            dist = np.zeros((new_tracks.shape[0], old_tracks.shape[0]))

            for i in range(new_tracks.shape[0]):
                for j in range(old_tracks.shape[0]):
                    tmp = lik.cluster_lik({new_tracks_idx[i], old_tracks_idx[j]})  # log space
                    dist[i, j] = tmp
            dist -= np.log(0.5) * num_sensors  # "divide" by likelihood for cluster size
            dist = -dist  # neg log space
            distance_th = 20

        # solve 2D assignment optimally
        new_idx, old_idx = linear_sum_assignment(dist)
        # add new tracks to corresponding clusters
        for n, o in zip(new_idx, old_idx):
            nt = new_tracks_idx[n]
            ot = old_tracks_idx[o]
            if dist[n, o] < distance_th:
                joint_asso[nt] = joint_asso[ot]
                clusters[joint_asso[nt]].append(nt)
            else:
                # create new cluster if assignment exceeds distance threshold
                joint_asso[nt] = num_clusters + 1
                clusters[joint_asso[nt]] = [nt]
                num_clusters += 1
        # create new clusters for tracks that have not been assigned
        remaining_new_tracks = set(np.arange(new_tracks_idx.shape[0])) - set(new_idx)
        for n in remaining_new_tracks:
            nt = new_tracks_idx[n]
            joint_asso[nt] = num_clusters + 1
            clusters[joint_asso[nt]] = [nt]
            num_clusters += 1

    return joint_asso
