# -*- coding:utf-8 -*-
import numpy as np
from scipy.stats import rankdata
import os

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='min', axis=1)
    filter_scores = scores * filters #
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)

def cal_performance(ranks,num):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    h_50 = sum(ranks <= 50) * 1.0 / len(ranks)

    def calculate_map(rank_list, K, num_list):
        if not len(rank_list) :
            return 0.0
        import numpy as np

        def calculate_ap(rankings, K):
            """
            Calculate AP@K for a single query.
            This function assumes that `rankings` is a list of the ranks of the relevant items.
            It sorts `rankings`, removes duplicates, and then computes the average precision.
            """
            unique_rankings = np.sort(np.unique(rankings))

            ap_sum = 0.0
            num_hits = 0
            total_relevant = len(unique_rankings)

            for i, rank in enumerate(unique_rankings):
                if rank <= K:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    ap_sum += precision_at_i

            return ap_sum / total_relevant if total_relevant > 0 else 0.0

        start_index = 0
        ap_values = []

        for num in num_list:
            # Extract the relevant ranks for this query
            relevant_ranks = rank_list[start_index:start_index + num]
            # Calculate AP for this query
            ap = calculate_ap(relevant_ranks, K)
            ap_values.append(ap)
            # Update the start index for the next query
            start_index += num

        # Calculate the mean of all AP values (MAP)
        mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0
        return mean_ap

    map_1 = calculate_map(ranks, 1,num)
    map_3 = calculate_map(ranks, 3,num)
    map_10 = calculate_map(ranks, 10,num)
    map_50 = calculate_map(ranks, 50,num)
    return mrr, h_1, h_3, h_10, h_50, map_1, map_3, map_10, map_50


def uniqueWithoutSort(a):
    indexes = np.unique(a, return_index=True)[1]
    res = [a[index] for index in sorted(indexes)]
    return res
