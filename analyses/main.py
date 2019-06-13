# klh 1/2019
import sys
print(sys.version) # ensure that you're using python3
import torch
print("PyTorch version = {}".format(torch.__version__))
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt
# %matplotlib notebook
import os
import pickle
import copy
import csv

# new since HW 2
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools

HIDDEN_SIZE = 15

# specify log file
logfile = '../logdirs/logdir_010/runlog_0.pkl'

# load results
with open(logfile, 'rb') as f:
    results = pickle.load(f)

# create dict whose keys are epochs and vals are test evaluation inds, for easy reference
epochs_to_inds = dict(zip([evaluation_event["enum"] for evaluation_event in results["test_data"]], 
                          range(len(results["test_data"]))))


def load_pickle(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d

def pull_item_activations(epoch_results, layer_name):
    # returns dict whose keys are item names and vals are mean activation patterns 
    # in the specified layer for corresponding to test set inputs containing that item
    item_labels = epoch_results["item_labels"]
    activations = epoch_results[layer_name]["act"]
    
    item_names = list(set(item_labels))
    out = dict(zip(item_names, [None for _ in range(len(item_names))]))
    for item in item_names:
        item_activations = [activations[i, :] for i in range(len(item_labels)) 
                                if item_labels[i] == item]
        if layer_name == "representation_layer":
            out[item] = item_activations[0] # representations are same across relations, so just pull first instance for this item
        else:
            out[item] = np.mean(np.stack(item_activations), 0)
    return out

def compute_mean_activation_pattern(item_names_to_activations, item_names):
    intersection_item_names = [name for name in item_names if name in 
            item_names_to_activations.keys()]
    try:
        return np.mean(np.stack([item_names_to_activations[item] for item in 
            intersection_item_names]), 0)
    except ValueError:
        return np.zeros(HIDDEN_SIZE)


# Euclidean distance between mean activation pattern for category 1 and mean activation pattern for category 2
def plot_pairwise_dists(results):
    categories_to_members = {"plants": ["plant", "tree", "flower", "pine", "oak", "rose", "daisy"],
                           "animals": ["animal", "bird", "fish", "robin", "canary", "sunfish", "salmon"],
                           "birds": ["bird", "canary", "robin"],
                           "fish": ["fish", "sunfish", "salmon"],
                           "trees": ["tree", "pine", "oak"],
                           "flowers": ["flower", "rose", "daisy"],
                           "robin": ["robin"],
                           "canary": ["canary"],
                           "pine": ["pine"],
                           "oak": ["oak"]}
    comparisons = [("plants", "animals"),
                   ("birds", "fish"),
                   ("trees", "flowers"),
                   ("robin", "canary"),
                   ("pine", "oak")]
    labels = ["epoch"] + comparisons
    
    dists = dict(zip(labels, [[] for _ in range(len(labels))]))
    
    for i in range(len(results["test_data"])):
        epoch = results["test_data"][i]["enum"]
        dists["epoch"].append(epoch)
        
        item_names_to_activations_this_epoch = pull_item_activations(results["test_data"][i], 
                                                                     "representation_layer")
        for comp in comparisons:            
            category1_mean_activations = compute_mean_activation_pattern(
                                        item_names_to_activations_this_epoch, 
                                        categories_to_members[comp[0]])
            category2_mean_activations = compute_mean_activation_pattern(
                                        item_names_to_activations_this_epoch, 
                                        categories_to_members[comp[1]])
            try:
                eucl_dist = spatial.distance.euclidean(category1_mean_activations,
                                                       category2_mean_activations)
            except ValueError:
                eucl_dist = 0
            
            dists[comp].append(eucl_dist)
            
    fig, ax = plt.subplots(figsize=[12,8])
    for comp in comparisons:
        ax.plot(dists["epoch"], dists[comp], label=comp)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Euclidean distance")
    ax.legend()
    ax.set_title("Distances between mean category representations")


def create_activation_bar_plots(epoch):
    r = results["test_data"][epochs_to_inds[epoch]] # pull data from epoch's evaluation event

    # pull activations
    item_names_to_activations = pull_item_activations(r, "representation_layer")
    
    # order labels to match Fig 4a
    ordered_labels = ["pine", "oak", "rose", "daisy", 
                      "robin", "canary", "sunfish", "salmon"]
    
    # create bar plots following order
    fig, axes = plt.subplots(len(ordered_labels), figsize=[8, 20])
    max_act = 0.
    for i in range(len(ordered_labels)):
        acts = item_names_to_activations[ordered_labels[i]]
        max_act = max(max(acts), max_act)
        axes[i].bar(range(len(acts)), acts)
        axes[i].set_ylim(0, max_act)
        axes[i].set_xticks([])
        axes[i].set_title(ordered_labels[i])

def create_dendrogram(epoch):
    r = results["test_data"][epochs_to_inds[epoch]] # pull data from epoch's evaluation event
    
    # pull activations
    item_names_to_activations = pull_item_activations(r, "representation_layer")
    # order labels to match Fig 4a
    labels = ["pine", "oak", "rose", "daisy",
                      "robin", "canary", "sunfish", "salmon"]
    
    # perform hierarchical clustering
    activations_as_array = np.stack({l: item_names_to_activations[l]
                                        for l in labels}.values())
    Z = linkage(activations_as_array, method="ward")

    # create dendrogram
    fig, ax = plt.subplots(figsize=[12,8])
    dn = dendrogram(Z, labels=labels) # scipy's dendrogram takes in labels and labels as appropriate
    ax.set_title("Epoch {}".format(epoch))


def svd():
    data_for_svd = load_pickle("data_for_svd.pickle")

    # {"items_by_attributes": items x attributes binary np array,
    #  "item_names": list of string names corresponding to rows,
    #  "attribute_names": list of string names corresponding to cols}

    # perform svd
    svd_input = data_for_svd["items_by_attributes"].transpose()
    u, s, vh = np.linalg.svd(svd_input, full_matrices=False)

    # plot
    attributes = data_for_svd["attribute_names"]
    items = data_for_svd["item_names"]

    fig, axes = plt.subplots(1, 4, figsize=[12,8])
    axes[0].imshow(svd_input, cmap="coolwarm")
    axes[0].set_title("Input-output correlation matrix")
    axes[0].set_xlabel("Items")
    axes[0].set_ylabel("Attributes")
    axes[0].set_yticks(range(len(attributes)))
    axes[0].set_yticklabels(attributes)

    axes[1].imshow(u, cmap="coolwarm")
    axes[1].set_title("U")
    axes[1].set_xlabel("Modes")
    axes[1].set_ylabel("Attributes")

    axes[2].imshow(np.diag(s), cmap="coolwarm")
    axes[2].set_title("S")
    axes[2].set_xlabel("Modes")
    axes[2].set_ylabel("Modes")

    # modes x items?
    axes[3].imshow(vh, cmap="coolwarm")
    axes[3].set_title("V^T")
    axes[3].set_xlabel("Items")
    axes[3].set_ylabel("Modes")
    axes[3].set_xticks(range(len(items)))
    axes[3].set_xticklabels(items)
    plt.xticks(rotation=90)

    # We could reconstruct the input-output correlation marix with np.matmul(np.matmul(u, np.diag(s)), vh)

#plot_pairwise_dists(results)
#create_activation_bar_plots(550)
#create_dendrogram(4000)
#svd()
#plt.savefig('00_3000_dendogram.png')
plt.show()
