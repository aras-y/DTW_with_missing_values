import numpy as np


def calculate_medoid_of_idxs(D, cluster_k_idxs):
    # Adapted from sklearn_extra.cluster.KMedoid.

    in_cluster_distances = D[
        cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
    ]

    # Calculate all costs from each point to all others in the cluster
    in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)
    min_cost_idx = np.argmin(in_cluster_all_costs)
    # min_cost = in_cluster_all_costs[min_cost_idx]
    return cluster_k_idxs[min_cost_idx]


def distortion_metric_precomputed(distance_matrix, labels):
    unique_labels = np.unique(labels)

    total_distortion = 0
    for label in unique_labels:


        cluster_k_idxs = np.where(labels == label)[0]
        medoid_idx = calculate_medoid_of_idxs(distance_matrix, cluster_k_idxs)
        distortion = np.sum(distance_matrix[cluster_k_idxs, medoid_idx])

        total_distortion += distortion
    return total_distortion
