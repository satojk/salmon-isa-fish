import csv
import random

import numpy as np
import torch

# utils
def read_in_csv(data_fname):
    # read in data such that each row is a list of dicts
    with open(data_fname, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        d = [] # list of dicts (one per training item)
        for line in reader:
            d.append(dict(zip(headers, [line[k] for k in headers])))
        return d
    
class Dataset(object):
    def __init__(self, data_fname="data.csv"):
        self.train_data = self.format_dataset(read_in_csv(data_fname))
        self.train_size = len(self.train_data["names"])
        self.test_data = self.train_data
        self.test_size = len(self.test_data["names"])
        
    def format_dataset(self, d_in):
        d = {"inputs": {"item": [], # list of items, where each item is a num_items one-hot tensor
                        "relation": []}, # list of relations as num_relations one-hot tensor
             "targets": [], # list of targets, where each target is a num_attribues binary tensor
             "names": [], # list of name strings, e.g. "pine isa"
             "item_names": [], # list of item name strings, e.g. "pine"
             "relation_names": []} # list of relation name strings, e.g. "isa"

        item_names = sorted(list(set([training_item["item"] for training_item in d_in]))) # alpha order
        relation_names = sorted(list(set([training_item["relation"] for training_item in d_in]))) # alpha order    
        attribute_names = [k for k in d_in[0].keys() if not k in ["item", "relation"]]
        self.num_items = len(item_names)
        self.num_relations = len(relation_names)
        self.num_attributes = len(attribute_names)

        # assign inds to names and create lookups in both directions for easy reference later
        self.item_names_to_inds = dict(zip(item_names, 
                                      list(range(self.num_items))))
        self.relation_names_to_inds = dict(zip(relation_names, 
                                          list(range(self.num_relations))))
        self.attribute_names_to_inds = dict(zip(attribute_names, 
                                           list(range(self.num_attributes))))

        self.item_inds_to_names = dict(zip(list(range(self.num_items)),
                                      item_names))
        self.relation_inds_to_names = dict(zip(list(range(self.num_relations)),
                                      relation_names))
        self.attribute_inds_to_names = dict(zip(list(range(self.num_attributes)),
                                      attribute_names))

        # package
        for training_item in d_in:
            d["inputs"]["item"].append(self.name_to_one_hot(
                                            training_item["item"], "item"))
            d["inputs"]["relation"].append(self.name_to_one_hot(
                                            training_item["relation"], "relation"))
            positive_attributes = []
            for k in self.attribute_names_to_inds.keys():
                if bool(int(training_item[k])):
                    positive_attributes.append(k)
            d["targets"].append(self.attributes_to_binary_pattern(positive_attributes))
            d["names"].append("{} {}".format(training_item["item"], # for visualization GUI only
                                             training_item["relation"]))
            d["item_names"].append(training_item["item"])
            d["relation_names"].append(training_item["relation"])
        return d
    
    def name_to_one_hot(self, name, input_type):
        assert input_type in ["item", "relation"]
        if input_type == "item":
            one_hot_size = self.num_items
            ind = self.item_names_to_inds[name]
        elif input_type == "relation":
            one_hot_size = self.num_relations
            ind = self.relation_names_to_inds[name]
        one_hot = torch.zeros(one_hot_size)
        one_hot[ind] = 1.
        return one_hot
            
    def one_hot_to_name(self, one_hot, input_type):
        assert input_type in ["item", "relation"]
        ind = int(torch.nonzero(one_hot).data.numpy())
        if input_type == "item":
            return self.item_inds_to_names[ind]
        elif input_type == "relation":
            return self.relation_inds_to_names[ind]
        
    def attributes_to_binary_pattern(self, attributes_as_names):
        # converts list of positive attributes into num_attributes-long tensor where 1. indicates 
        # attribute is present and 0. indicates that it's absent
        out = torch.zeros(self.num_attributes)
        for name in attributes_as_names:
            out[self.attribute_names_to_inds[name]] = 1.
        return out
    
    def attribute_pattern_to_names(self, attribute_pattern):
        # returns names of positive attributes given binary pattern tensor
        inds = torch.nonzero(attribute_pattern)
        if inds.shape[0] == 1:
            inds = list(inds[0].data.numpy())
        else:
            inds = list(inds.squeeze().data.numpy())
        return [self.attribute_inds_to_names[i] for i in inds]
    
    def print_batch(self, batch_inputs, batch_targets, predictions=None, 
                    targets_as_names=False):
        for batch_item_ind in range(len(batch_inputs["item"])):
            print("\n")
            print("Input: {}, {}".format(
                    self.one_hot_to_name(batch_inputs["item"][batch_item_ind], "item"),
                    self.one_hot_to_name(batch_inputs["relation"][batch_item_ind], "relation")))
            if targets_as_names:
                print(batch_targets[batch_item_ind])
                print("Target: {}".format(self.attribute_pattern_to_names(
                                              batch_targets[batch_item_ind])))
            else:
                print("Target: {}".format(batch_targets[batch_item_ind].data.numpy()))
            if not predictions is None:
                print("Model output: {}".format(predictions[batch_item_ind, 
                    :].data.numpy()))
    
    def prepare_batch(self, batch, batch_size):
        # stack batch items to create batch_size x num_items, batch_num_relations, 
        # and batch_size x num_attributes tensors
        inputs = {"item": torch.stack(batch["inputs"]["item"], 0),
                  "relation": torch.stack(batch["inputs"]["relation"], 0)}
        targets = torch.stack(batch["targets"], 0)
        return (inputs, targets)
            
    def get_per_epoch_batches(self, batch_size):
        assert self.train_size % batch_size == 0
        train_item_inds = list(range(self.train_size))
        random.shuffle(train_item_inds)
        batches = []
        
        for batch_start in np.arange(0, self.train_size, batch_size):
            batch_inds = train_item_inds[batch_start : batch_start + batch_size]
            # pull selected inds from train set
            batch = {"inputs": 
                        {"item": [self.train_data["inputs"]["item"][i] for i in batch_inds], 
                         "relation": [self.train_data["inputs"]["relation"][i] for i in batch_inds]},
                     "targets": [self.train_data["targets"][i] for i in batch_inds]}
            batches.append(self.prepare_batch(batch, batch_size)) # reformat to be tensors
        return batches
