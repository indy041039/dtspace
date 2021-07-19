import pandas as pd
import numpy as np

def plot_decision_path(model, X):
    feature = model.tree_.feature
    node_indicator = model.decision_path(X)
    leaf_id = model.apply(X)
    features_name = X.columns
    X = X.values
    size = len(X)
    result = []
    for sample_id in tqdm(range(size)):
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
        decision_path_list = []
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue
            decision_path_list.append(features_name[feature[node_id]])
        result.append(list(set(decision_path_list)))
    return result
