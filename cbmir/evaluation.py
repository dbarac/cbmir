import time
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset

df = dataset.train_df

# possible references for evaluation: body_part, modality or both
body_parts = list(df["BodyPartExaminedFixed"].unique())
body_parts.sort()

modalities = list(df["Modality"].unique())
modalities.sort()

# correct class <-> both the correct body_part and modality was guessed
bps_and_mods = [(bp, mod) for bp in body_parts for mod in modalities]

def encode_label(instance_category, categories):
    assert type(categories) == list
    """
    Encode as integer from 0 to len(classes)-1.
    Categories should be sorted alphabetically.
    """
    return categories.index(instance_category)

def mean_precision(true_cat, nn_cats, k_max):
    """
    Calculate mean precision for k=1,...,k_max.

    Mean precision evaluation for given k:
    1. calculate precision for k nearest neighbours of each query image:
       (number of retrieved images with the same class as the query image) / k
    2. calculate the mean of Precision scores of all images in the test dataset.
    """
    assert true_cat.ndim == 2 and true_cat.shape[1] == 1

    mean_precisions = []
    for k in range(1, k_max+1):
        precision = (true_cat == nn_cats[:, :k]).sum(axis=1) / k
        mean_precisions.append(precision.mean())
    return mean_precisions

def mean_avg_precision(true_cat, nn_cats, k_max):
    """
    Calculate mean Average Precision.

    Mean Average Precision for given k_max:
    1. calculate precision for k nearest neighbours of each query image (k=1,...,k_max):
       (number of retrieved images with the same class as the query image) / k
    2. calculate mean of precisions for each k=1,..,k_max (average precision)
    3. calculate the mean of Average Precision scores of all images in the test dataset.
    """
    assert true_cat.ndim == 2 and true_cat.shape[1] == 1

    mean_precisions = []
    precisions = np.empty((len(true_cat), k_max))
    for k in range(1, k_max+1):
        precisions[:, k-1] = (true_cat == nn_cats[:, :k]).sum(axis=1) / k
    avg_precisions = precisions.mean(axis=1)
    mean_avg_precision = avg_precisions.mean()
    return mean_avg_precision

def evaluate(train_df, test_df, train_idxs, test_idxs, retreived_nns, max_k):
    # retrieval can be evaluated by referring to 3 image properties:
    # body part examined, modality or both at once
    num_references = 3 
    BODY_PART = 0
    MODALITY = 1
    BOTH = 2

    true_cat = np.zeros((num_references, len(test_idxs), 1))
    nn_cats = np.zeros((num_references, len(test_idxs), max_k))

    for i, img_idx in enumerate(test_idxs):
        row = test_df.iloc[img_idx]

        body_part = row["BodyPartExaminedFixed"]
        true_cat[BODY_PART, i] = encode_label(body_part, body_parts)

        modality = row["Modality"]
        true_cat[MODALITY, i] = encode_label(modality, modalities)

        bp_and_mod = (body_part, modality)
        true_cat[BOTH, i] = encode_label(bp_and_mod, bps_and_mods)

        for k, retreived_idx in enumerate(retreived_nns[i]):
            row = train_df.iloc[retreived_idx]

            body_part = row["BodyPartExaminedFixed"]
            nn_cats[BODY_PART, i, k] = encode_label(body_part, body_parts)

            modality = row["Modality"]
            nn_cats[MODALITY, i, k] = encode_label(modality, modalities)

            bp_and_mod = (body_part, modality)
            nn_cats[BOTH, i, k] = encode_label(bp_and_mod, bps_and_mods)

    mAP = [
        mean_avg_precision(true_cat[BODY_PART], nn_cats[BODY_PART], max_k),
        mean_avg_precision(true_cat[MODALITY], nn_cats[MODALITY], max_k),
        mean_avg_precision(true_cat[BOTH], nn_cats[BOTH], max_k)
    ]
    print("mean Average Precision (mAP) [body_part, modality, both]:", mAP)

    # mP
    results = []
    print("Mean precision (Body part examined) for top-k from 1 to", str(max_k) + ":")
    mean_precisions_per_k = mean_precision(true_cat[BODY_PART], nn_cats[BODY_PART], max_k)
    print(mean_precisions_per_k)
    results.append(mean_precisions_per_k)

    print("Mean precision (Modality) for top-k from 1 to", str(max_k) + ":")
    mean_precisions_per_k = mean_precision(true_cat[MODALITY], nn_cats[MODALITY], max_k)
    print(mean_precisions_per_k)
    results.append(mean_precisions_per_k)

    print("Mean precision (Body part examined and Modality) for top-k from 1 to", str(max_k) + ":")
    mean_precisions_per_k = mean_precision(true_cat[BOTH], nn_cats[BOTH], max_k)
    print(mean_precisions_per_k)
    results.append(mean_precisions_per_k)

    return results

def plot_model_comparison(labels, results, title, k, save_path, y_lim=[0.5, 1]):
    """
    Compare models or hyperparameter combinations by
    plotting mean precision for different k values.
    """
    assert len(labels) == len(results)

    k_vals = np.arange(1, k+1)
    markers = itertools.cycle(['+', '*', 'o', '1', 'p'])
    linestyles = itertools.cycle(['--', '-.', '-', ':'])
    axes = plt.gca()
    axes.set_ylim(y_lim)
    figure = plt.gcf()
    figure.set_size_inches(8, 5)
    plt.rcParams.update({'font.size': 12})

    for label, y_vals in zip(labels, results):
        plt.plot(k_vals, y_vals, linestyle=next(linestyles), marker=next(markers),
                 markersize=12, linewidth=2, label=label)
    plt.legend(loc="lower left")
    plt.ylabel("Mean top-k precision (mP@k)", fontsize=13)
    plt.xlabel("k", fontsize=13)
    plt.xticks(k_vals, fontsize=13)
    plt.grid(linestyle='--')

    y_lim[1] += 0.025 # to make sure top limit is included
    plt.yticks(np.arange(*y_lim, step=0.025), fontsize=13)
    if title:
        plt.title(title)
    plt.savefig(save_path + ".pdf", dpi=100, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=100, bbox_inches='tight')
    plt.show()
