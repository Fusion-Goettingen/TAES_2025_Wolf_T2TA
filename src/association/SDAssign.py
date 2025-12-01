import matlab.engine
import numpy as np
from src.association.Likelihood import Likelihood


class SDAssign:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

    def associate_sd(self, tracks, detection_prob, spatial_sig, likelihood):
        sensors, sensor_counts = np.unique(tracks[:, -1], return_counts=True)

        num_sensors = sensors.shape[0]

        lik = Likelihood("const", spatial_sig, likelihood)
        lik.set_tracks(tracks, num_sensors, p_D=detection_prob)

        shape = sensor_counts + 1
        cost_matrix = np.ones(shape) * -1e5

        sensor_dict = {s: np.where(tracks[:, -1] == s)[0] for s in sensors}

        # Precompute individual track likelihoods
        track_liks = {t: lik.cluster_lik({t}) for t in range(tracks.shape[0])}

        # compute cost matrix
        for index in np.ndindex(cost_matrix.shape):
            if np.sum(index) == 0:
                continue

            track_set = [sensor_dict[sensors[s]][index[s] - 1] for s in range(num_sensors) if index[s] > 0]

            cost_matrix[index] = lik.cluster_lik(set(track_set))  # log space
            cost_matrix[index] -= sum(track_liks[t] for t in track_set)  # divide by every track is a singleton

        cost_matrix = -cost_matrix  # neg log space

        # compute assignmend (matlab algorithm)
        assignments, cost, gap = self.eng.assignsd(cost_matrix, 0.01, 200, nargout=3)

        # reconstruct joint association
        assignments = np.array(assignments) - 1

        joint_asso = np.zeros(tracks.shape[0], dtype=np.int64)

        for a, assign in enumerate(assignments):
            track_set = [sensor_dict[sensors[s]][assign[s] - 1] for s in range(num_sensors) if assign[s] > 0]
            joint_asso[track_set] = a + 1

        return joint_asso
