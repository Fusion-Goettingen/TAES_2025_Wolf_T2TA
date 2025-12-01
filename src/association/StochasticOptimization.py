import numpy as np
from src.association.Likelihood import Likelihood


class T2TA_SO:
    def __init__(
            self,
            pd_method,
            max_pd=None,
            min_pd=None,
            else_pd=None,
            max_dist=None,
            pd_est_factor=2,
            spatial_sig=2,
            likelihood="ml_const",
    ):
        self.gating_th = spatial_sig * 3 * 2  # diameter of 3 sigma bound

        self.cov = None
        self.cov_inv = None
        self.cov_det = None

        self.likelihood = Likelihood(
            pd_method, max_pd, min_pd, else_pd, max_dist, pd_est_factor, spatial_sig, likelihood, cache=True
        )
        self.likelihood_select = Likelihood(
            pd_method, max_pd, min_pd, else_pd, max_dist, pd_est_factor, spatial_sig, likelihood, cache=True
        )

    def associate(
            self,
            tracks,
            num_sweeps,
            num_sensors,
            tracks_df=None,
            sensor_pos=None,
            return_best=True,
            return_best_num=1,
            p_D=None,
    ):
        # max pD used for sampling, use correct pD for selection
        if p_D is not None and p_D > 0.97:
            self.likelihood.set_tracks(tracks, num_sensors, tracks_df, sensor_pos, 0.97)
            self.likelihood_select.set_tracks(tracks, num_sensors, tracks_df, sensor_pos, p_D)
        else:
            self.likelihood.set_tracks(tracks, num_sensors, tracks_df, sensor_pos, p_D)
            self.likelihood_select.set_tracks(tracks, num_sensors, tracks_df, sensor_pos, p_D)
        num_tracks = tracks.shape[0]

        # initialize first sample and clusters
        curr_sample = np.arange(num_tracks) + 1  # all tracks are singletons
        next_cluster = num_tracks
        clusters = {}
        for i in range(num_tracks):
            clusters[i + 1] = {
                "tracks": {i},
                "lik": self.likelihood.cluster_lik({i}),
                "mean": tracks[i, :-1],
            }

        samples = np.zeros((num_sweeps * num_tracks, num_tracks), dtype=np.int64)
        weights = np.zeros(num_sweeps * num_tracks)

        counter = 0
        saved_samples_list = []

        if num_sweeps == 0:
            return

        for n in range(num_sweeps):
            for t in range(num_tracks):
                # compute sampling likelihood
                curr_cluster = clusters[curr_sample[t]]

                num_clusters = len(clusters)

                sample_lik = np.ones(2 * num_clusters + 2) * -1e5  # log space

                weight_curr_cluster = curr_cluster["lik"]
                weight_curr_cluster_minus_t = self.likelihood.cluster_lik(
                    curr_cluster["tracks"] - {t},
                )

                # remain in current cluster
                sample_lik[0] = 0.0  # log space

                # extract singleton
                if len(curr_cluster["tracks"]) > 1:
                    sample_lik[1] = (
                            self.likelihood.cluster_lik({t}) + weight_curr_cluster_minus_t - weight_curr_cluster
                    )  # log space

                clusters_sorted = sorted(clusters)
                dist_to_clusters = np.linalg.norm(
                    np.concatenate([clusters[c]["mean"][None, :] for c in clusters_sorted], axis=0)
                    - tracks[t, :-1],
                    axis=1,
                )
                clusters_possible = np.nonzero(dist_to_clusters < self.gating_th)[0]
                # iterate over all other clusters
                for i in clusters_possible:
                    c = clusters_sorted[i]
                    if c == curr_sample[t]:  # current cluster
                        continue

                    weight_cluster_c = clusters[c]["lik"]
                    # add current track to cluster
                    if tracks[t, -1] not in tracks[list(clusters[c]["tracks"]), -1]:
                        sample_lik[i + 2] = (
                                self.likelihood.cluster_lik(clusters[c]["tracks"] | {t})
                                + weight_curr_cluster_minus_t
                                - (weight_curr_cluster + weight_cluster_c)
                        )  # log space

                    # merge clusters if current cluster is >1 and sensors are disjoint
                    if len(curr_cluster["tracks"]) > 1 and set(tracks[list(curr_cluster["tracks"]), -1]).isdisjoint(
                            set(tracks[list(clusters[c]["tracks"]), -1])
                    ):
                        # log space
                        sample_lik[num_clusters + i + 2] = self.likelihood.cluster_lik(
                            clusters[c]["tracks"] | curr_cluster["tracks"]
                        ) - (weight_curr_cluster + weight_cluster_c)

                sample_lik = np.exp(sample_lik)  # log space
                # not in log space anymore, normalize
                sample_lik /= np.sum(sample_lik)

                # sample
                random_samples = np.random.random(sample_lik.size)
                assign = np.argmax(random_samples * sample_lik)

                # perform action
                # singleton and current cluster >1
                if assign == 1 and len(curr_cluster["tracks"]) > 1:

                    curr_cluster["tracks"] -= {t}
                    curr_cluster["lik"] = self.likelihood.cluster_lik(curr_cluster["tracks"])
                    curr_cluster["mean"] = tracks[list(curr_cluster["tracks"]), :-1].mean(axis=0)
                    clusters[next_cluster] = {
                        "tracks": {t},
                        "lik": self.likelihood.cluster_lik({t}),
                        "mean": tracks[t, :-1],
                    }
                    curr_sample[t] = next_cluster
                    next_cluster += 1

                # move track
                elif 1 < assign <= num_clusters + 1:

                    assign_c = sorted(clusters.keys())[assign - 2]

                    curr_cluster["tracks"] -= {t}
                    curr_cluster["lik"] = self.likelihood.cluster_lik(curr_cluster["tracks"])

                    if len(curr_cluster["tracks"]) == 0:
                        del clusters[curr_sample[t]]
                    elif len(curr_cluster["tracks"]) > 1:
                        curr_cluster["mean"] = tracks[list(curr_cluster["tracks"]), :-1].mean(axis=0)
                    elif len(curr_cluster["tracks"]) == 1:
                        curr_cluster["mean"] = tracks[list(curr_cluster["tracks"])[0], :-1]

                    clusters[assign_c]["tracks"] |= {t}
                    clusters[assign_c]["lik"] = self.likelihood.cluster_lik(clusters[assign_c]["tracks"])
                    clusters[assign_c]["mean"] = tracks[list(clusters[assign_c]["tracks"]), :-1].mean(axis=0)

                    curr_sample[t] = assign_c

                # merge clusters
                elif num_clusters + 1 < assign:

                    assign_c = sorted(clusters.keys())[assign - 2 - num_clusters]
                    old_cluster = curr_sample[t]

                    curr_sample[list(curr_cluster["tracks"])] = assign_c
                    clusters[assign_c]["tracks"] |= clusters[old_cluster]["tracks"]
                    clusters[assign_c]["lik"] = self.likelihood.cluster_lik(clusters[assign_c]["tracks"])
                    clusters[assign_c]["mean"] = tracks[list(clusters[assign_c]["tracks"]), :-1].mean(axis=0)

                    del clusters[old_cluster]

                # check if association has been saved before
                curr_sample, idx_map = sanitize_association(curr_sample)
                clusters = {idx_map[c]: clusters[c] for c in clusters}
                if not np.all(curr_sample == samples[:counter], axis=1).any():
                    samples[counter] = curr_sample.copy()
                    weights[counter] = np.sum([clusters[c]["lik"] for c in clusters])  # log space
                    saved_samples_list.append(n * num_tracks + t)

                    counter += 1

        # select associations
        weights_orig_pd = self.likelihood_select.compute_weights(samples[:counter])
        if return_best:
            if return_best_num == 1:
                return samples[np.argmax(weights_orig_pd[:counter])], weights_orig_pd[:counter].max()
            else:
                idx = weights_orig_pd.argsort()[-return_best_num:][
                      ::-1]  # highest likelihoods in the end, reverse order
                return samples[idx], weights_orig_pd[idx]
        else:
            return (
                samples[:counter],
                weights_orig_pd[:counter],
                saved_samples_list,
            )


def sanitize_association(sample):
    cluster_unique, index, inverse = np.unique(sample, return_index=True, return_inverse=True)
    cluster_order = cluster_unique[np.argsort(index)]  # unique clusters in order they appear
    new_clusters = np.arange(cluster_order.shape[0]) + 1  # rename clusters in ascending order
    new_sample = new_clusters[np.argsort(cluster_order)][inverse]  # reconstruct array

    idx_map = {old: new for new, old in zip(new_clusters, cluster_order)}

    return new_sample, idx_map
