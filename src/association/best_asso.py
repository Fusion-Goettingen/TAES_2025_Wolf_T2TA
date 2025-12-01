import numpy as np


# ground truth association in artery scenario
def gt_association(tracks):
    n = len(tracks)
    gt_asso = np.zeros(n, dtype=np.int64)

    id_map = {}
    cluster_count = 1

    for i in range(n):
        ext_id = tracks['external_id'].at[tracks.index[i]]

        if ext_id not in id_map:
            id_map[ext_id] = cluster_count
            cluster_count += 1

        gt_asso[i] = id_map[ext_id]
    return gt_asso
