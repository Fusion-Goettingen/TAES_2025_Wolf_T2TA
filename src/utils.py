import os
import pickle

import numpy as np
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

from src.abspath import root_folder


def compute_gospa(objects, tracks, association):
    num_objects = objects.shape[0]
    cluster_ids = set(association)
    cluster_center = []
    for c in cluster_ids:
        if c == 0:
            for t in tracks[association == c]:
                cluster_center.append(State(StateVector(t[:-1])))
        else:
            cluster_center.append(State(StateVector(np.mean(tracks[association == c, :-1], axis=0))))

    objects = [State(StateVector(o)) for o in objects]
    gospa = GOSPAMetric(p=1, c=10).compute_gospa_metric(cluster_center, objects)[0].value

    return {**gospa, **{k + "_po": gospa[k] / num_objects for k in gospa}}



def sort_gospa(x, num_sweeps):
    num_tracks = x.best_asso.shape[0]
    max_samples = num_sweeps * num_tracks
    if "SO" not in x.method:
        return x.gospa
    else:
        max_idx = [np.argmax(x["weights"][: n + 1]) for n in range(x["weights"].shape[0])]
        idx_ext = np.zeros(max_samples, dtype=np.int64)
        tmp = x.saved_samples + [max_samples]
        for i in range(len(x.saved_samples)):
            idx_ext[tmp[i] : tmp[i + 1]] = max_idx[i]
        idx_ext = [idx_ext[(i + 1) * num_tracks - 1] for i in range(num_sweeps)]
        return x["gospa_batch"][idx_ext]
