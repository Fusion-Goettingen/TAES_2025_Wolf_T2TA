import numpy as np


class Likelihood:
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
            cache=False,
    ):
        self.pd_method = pd_method
        self.likelihood = likelihood
        self.spatial_sig = spatial_sig

        if self.pd_method == "const":
            self.detection_prob = None
            self.binom_prob = []
            self.pd_est_factor = pd_est_factor
        if self.pd_method == "sensor_dist":
            self.sensor_pos = None
            self.max_pd = max_pd
            self.min_pd = min_pd
            self.else_pd = else_pd
            self.max_dist = max_dist
        self.tracks = None
        self.tracks_df = None
        self.sensors = None
        self.num_sensors = None

        self.cov = None
        self.cov_inv = None
        self.cov_det = None
        self.means = None
        self.covs_inv = None
        self.covs_det = None
        self.do_cache = cache
        if self.do_cache:
            self.cache = {}

    def set_tracks(self, tracks, num_sensors, tracks_df=None, sensor_pos=None, p_D=None):
        self.tracks = tracks
        self.tracks_df = tracks_df

        if self.tracks_df is not None:
            # extract states and covariances from dataframe
            self.means = np.array(self.tracks_df["mean"].to_list())
            self.covs = np.array(self.tracks_df["cov"].to_list())
            self.covs_inv = np.array(self.tracks_df["cov_inv"].to_list())
            self.covs_det = np.array(self.tracks_df["cov_det"].to_list())

        num_tracks = tracks.shape[0]
        self.num_sensors = num_sensors

        self.sensors = np.unique(tracks[:, -1])
        num_tracks_per_sensor = np.zeros_like(self.sensors)

        if self.pd_method == "const":
            if p_D is None:
                # estimate constant pD
                for i, s in enumerate(self.sensors):
                    n = np.sum(tracks[:, -1] == s)
                    num_tracks_per_sensor[i] = n
                self.detection_prob = num_tracks / (self.pd_est_factor * num_tracks_per_sensor.max() * num_sensors)
            else:
                self.detection_prob = p_D

            undetected_prob = 1 - self.detection_prob

            # precompute likelihood of cluster cardinality
            self.binom_prob = [
                self.detection_prob ** i * undetected_prob ** (num_sensors - i) for i in range(num_sensors + 1)
            ]
        elif self.pd_method == "sensor_dist":
            self.sensor_pos = sensor_pos
            self.sensors = np.unique(self.sensor_pos[:, -1])

    def cluster_lik(self, cluster):
        """
        computes the likelihood of one cluster
        """
        w = 0.0  # log space
        if cluster == set():
            return w

        if self.do_cache:
            cluster_frozen = frozenset(cluster)
            if cluster_frozen in self.cache:
                return self.cache[cluster_frozen]
        idx = list(cluster)
        c_len = len(cluster)

        # spatial likelihood
        cluster_tracks = self.tracks[idx, :-1]
        mean = np.mean(cluster_tracks, axis=0)

        if self.likelihood == "ml_const":
            cov = np.eye(mean.shape[0]) * self.spatial_sig ** 2 * (1 + 1 / c_len)

            prob = gaussian_likelihood(
                mean,
                np.linalg.inv(cov),
                np.linalg.det(cov),
                cluster_tracks,
            )

            w += np.sum(np.log(prob + 1e-300))  # log space

        elif self.likelihood == "ml":
            means = self.means[idx]
            covs_inv = self.covs_inv[idx]
            covs = self.covs[idx]

            cov_gl = np.linalg.inv(covs_inv.sum(axis=0))
            mean = (cov_gl @ (covs_inv @ means[:, :, None]).sum(axis=0)).transpose().squeeze()
            covs_inv_ml = np.linalg.inv(covs + cov_gl)
            covs_det_ml = np.linalg.det(covs + cov_gl)

            prob = gaussian_likelihood(
                mean,
                covs_inv_ml,
                covs_det_ml,
                means,
            )

            w += np.sum(np.log(prob + 1e-300))  # log space

        elif self.likelihood == "gl_const":
            if self.cov is None:
                self.cov = np.eye(mean.shape[0]) * self.spatial_sig ** 2
                self.cov_inv = np.linalg.inv(self.cov)
                self.cov_det = np.linalg.det(self.cov)
            prob = gaussian_likelihood(
                mean,
                self.cov_inv,
                self.cov_det,
                cluster_tracks,
            )

            w += np.sum(np.log(prob + 1e-300))  # log space
        elif self.likelihood == "gl":
            means = self.means[idx]
            covs_inv = self.covs_inv[idx]
            covs_det = self.covs_det[idx]

            cov_gl = np.linalg.inv(covs_inv.sum(axis=0))
            mean = (cov_gl @ (covs_inv @ means[:, :, None]).sum(axis=0)).transpose().squeeze()
            prob = gaussian_likelihood(
                mean,
                covs_inv,
                covs_det,
                means,
            )

            w += np.sum(np.log(prob + 1e-300))  # log space

        # likelihood for cluster size
        if self.pd_method == "const":
            w += np.log(self.binom_prob[c_len] + 1e-300)  # log space

        elif self.pd_method == "sensor_dist":
            pds = self.compute_pd_sensor_dist(mean)
            w += np.sum(
                np.log(
                    np.where(
                        np.isin(self.sensors, self.tracks[idx, -1]),
                        pds,
                        1 - pds,
                    )
                    + 1e-300
                )
            )
        if self.do_cache:
            self.cache[cluster_frozen] = w
        return w

    def compute_weights(self, samples):
        """
        compute likelihoods of several joint associations
        """
        if len(samples.shape) == 1:
            samples = samples[None, :]

        weights = np.zeros(samples.shape[0])  # log space
        for i, s in enumerate(samples):
            for c in set(s):
                idx = set(np.where(s == c)[0])
                weights[i] += self.cluster_lik(idx)  # log space

        return weights

    def compute_pd_sensor_dist(self, mean):
        """
        compute pD based on distance to sensor
        """
        dists = np.linalg.norm(mean[:2] - self.sensor_pos[:, :2], axis=1)
        pds = ((-self.max_pd + self.min_pd) / self.max_dist) * dists + self.max_pd
        pds[dists > self.max_dist] = self.else_pd
        pds[pds < self.else_pd] = self.else_pd
        return pds


def gaussian_likelihood(mean, cov_inv, cov_det, x):
    if len(x.shape) == 1:
        x = x[None, :]
    lik = (
            (2 * np.pi) ** (-x.shape[1] / 2)
            * cov_det ** (-0.5)
            * np.exp(-0.5 * (x - mean)[:, None, :] @ cov_inv @ (x - mean)[:, :, None]).squeeze()
    )
    return np.squeeze(lik)
