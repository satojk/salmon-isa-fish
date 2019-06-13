import os
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# utils
def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def get_logdir():
    results_dir = "logdirs"
    prefix = "logdir"
    fs = sorted([int(f.split("_")[1]) for f in os.listdir(results_dir) if prefix in f])
    if len(fs) == 0:
        num = 0
    else:
        num = fs[-1] + 1
    new_f = os.path.join(results_dir, prefix + "_" + str(num).zfill(3))
    return new_f

class Trainer(object):
    def __init__(self, dataset, model,
                 name="",
                 train_batch_size=32,
                 num_training_epochs=2000, 
                 stopping_criterion=0.04,
                 learning_rate=1e-04,
                 save_freq=1, test_freq=10,
                 checkpoint_freq=200, 
                 print_freq=0, show_plot=False,
                 save_dir=None,
                 new_items=None,
                 representation_units_to_add=0):
        self.new_items = new_items
        self.parent_items = {
                "flower": "plant",
                "tree": "plant",
                "bird": "animal",
                "fish": "animal",
                "daisy": "flower",
                "rose": "flower",
                "oak": "tree",
                "pine": "tree",
                "robin": "bird",
                "canary": "bird",
                "sunfish": "fish",
                "salmon": "fish"
            }
        self.dataset = dataset
        self.model = model
        self.name = name
        
        # training parameters
        self.train_batch_size = train_batch_size
        assert self.dataset.train_size % self.train_batch_size == 0
        self.num_training_epochs = num_training_epochs
        self.stopping_criterion = stopping_criterion
        self.learning_rate = learning_rate
        
        # loss function: sum across batch items of sum of squared errors across tasks
        self.criterion = nn.MSELoss(reduction="sum") # summed squared error summed across batch items
        
        # optimization: gradient descent
        self.optimizer = optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                    lr=self.learning_rate, momentum=0.)
        
        # viewing & saving parameters
        self.save_dir = save_dir
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.test_freq = test_freq
        self.checkpoint_freq = checkpoint_freq
        self.show_plot = show_plot
        self.setup()
        self.inherit_weights()
        self.add_units(representation_units_to_add)

    def inherit_weights(self):
        for new_item in self.new_items:
            print("Inheriting weight for {} from {}".format(new_item,
                self.parent_items[new_item]))
            self.model.copy_weights_to_rep(
                    self.dataset.item_names_to_inds[self.parent_items[new_item]],
                    self.dataset.item_names_to_inds[new_item])

    def add_units(self, representation_units_to_add):
        if representation_units_to_add:
            self.model.add_representation_units(representation_units_to_add)
        
    def setup(self):
        if self.save_dir is None:
            self.save_dir = os.path.join(get_logdir())
        self.checkpoints_dir = os.path.join(self.save_dir, "checkpoint_files_{}".format(self.name))
        [make_dir(p) for p in [self.save_dir, self.checkpoints_dir]]
        self.log_file = os.path.join(os.getcwd(), self.save_dir, "runlog_{}.pkl".format(self.name))
        self.checkpoint_file = os.path.join(self.checkpoints_dir, "checkpoint.pth".format(self.name))
        
        # load checkpoint if already exists
        if os.path.exists(self.checkpoint_file):
            print("Loading from checkpoint: {}".format(self.checkpoint_file))
            checkpoint = torch.load(self.checkpoint_file)
            self.start_epoch = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.load_results()
        else:
            self.start_epoch = 0
            self.loss_data = {"epochs": [], "train_losses": [], "train_accs": []}
            self.log = []
            
    def save_pickle(self, d, fname):
        with open(fname, 'wb') as f:
            pickle.dump(d, f)

    def load_pickle(self, fname):
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        return d

    def save_results(self):
        info = {"test_data": self.log,
                "loss_data": self.loss_data}
        self.save_pickle(info, self.log_file)

    def load_results(self):
        f = self.load_pickle(self.log_file)
        self.log = f["test_data"]
        self.loss_data = f["loss_data"]
        
    def save_checkpoint(self, epoch):
        checkpoint = {"epoch": epoch,
                      "model_state": self.model.state_dict(),
                      "optimizer_state": self.optimizer.state_dict()}
        torch.save(checkpoint, self.checkpoint_file)
        
    def final_save_and_view(self, epoch, loss):
        self.eval_test_points(epoch + 1)
        self.print_final()
        if self.show_plot:
            self.plot_loss()
            self.plot_acc()
        self.save_results()
        self.save_checkpoint(epoch)

    def init_info_dict(self, epoch, layer_names):
        formatted = self.dataset.prepare_batch(self.dataset.test_data, self.dataset.test_size)
        test_inputs, test_targets = formatted[0], formatted[1]
        
        info = {"enum": epoch,
                "input": torch.cat(
                            (test_inputs["item"], 
                             test_inputs["relation"]), 1).data.numpy(),
                "target": test_targets.data.numpy(),
                "labels": self.dataset.test_data["names"],
                "item_labels": self.dataset.test_data["item_names"],
                "relation_labels": self.dataset.test_data["relation_names"],
                "loss_sum": 0., # summed loss over batch items
                "loss": np.zeros(self.dataset.test_size)} # losses each batch items (batch_size-long)
        
        for i in range(len(self.model.linear_layers)):
            layer_info = {"input_": None, # layer input (batch_sz x layer_inp_sz), as recorded in self.model.layer_inputs
                          "weights": 
                          copy.deepcopy(self.model.linear_layers[i].weight.data.numpy()),
                          "biases": 
                          copy.deepcopy(self.model.linear_layers[i].bias.data.numpy()),
                          "net": None, # output of linear layer, before nonlinearity (batch_sz x layer_output_sz), as recorded in self.model.layer_outputs
                          "act": None, # layer activations (batch_sz x layer_output_sz), as recorded in self.model.layer_activations
                          "gweights": None, # dE/dW (batch_sz x weight_sz[0] x weight_sz[1])
                          "gbiases": None, # dE/dB (batch_sz x bias_sz[0])
                          "gnet": None, #dE/dnet (batch_sz x layer_output_sz)
                          "gact": None, #dE/da (batch_sz x layer_output_sz)
                          "sgweights": None, # gweights summed across test items (weight_sz[0] x weight_sz[1])
                          "sgbiases": None} # gbiases summed across test items (bias_sz-long)
            info[layer_names[i]] = layer_info
        return info

    def eval_test_points(self, epoch, print_info=False, print_predictions=True):
        # record gradients w.r.t. each datapoint for visualization purposes only
        info = self.init_info_dict(epoch, self.model.layer_names)  
    
        # enumerate the test data so that we can view the gradients 
        # for each test point individually. Normally, you will want to feed
        # the test data through the model as a single batch.
        for i in range(self.dataset.test_size):
            test_batch = self.dataset.prepare_batch(
                        {"inputs": 
                              {"item": [self.dataset.test_data["inputs"]["item"][i]], 
                               "relation": [self.dataset.test_data["inputs"]["relation"][i]]},
                          "targets": [self.dataset.test_data["targets"][i]]}, 1)
            test_input, test_target = test_batch[0], test_batch[1]

            self.optimizer.zero_grad()
            test_prediction = self.model.forward(test_input, record_data=True)
            test_loss = self.criterion(test_prediction, test_target)
            
            # Here, we call the backward method in order to compute gradients for
            # visualization purposes. In your models, do NOT do this during test
            # evaluation! It can lead to training on the test set. 
            #test_loss.backward()

            #info["loss"][i] = test_loss.data.numpy()
            #info["loss_sum"] += test_loss.data.numpy()
            
            # reverse to correspond to the forward direction
            #self.model.gnet.reverse()
            #self.model.gact.reverse()

            # update info dict
            for j in range(len(self.model.linear_layers)):
                layer_name = self.model.layer_names[j]
                
                def update(k, new_v, expand=False):
                    new_v = copy.deepcopy(new_v.data.numpy())
                    if expand == True:
                        new_v = np.expand_dims(new_v, 0)
                    if i == 0:
                        info[layer_name][k] = new_v
                    else:
                        info[layer_name][k] = np.vstack((info[layer_name][k], new_v))
                
                update("input_", self.model.layer_inputs[j])
                update("net", self.model.layer_outputs[j])
                update("act", self.model.layer_activations[j])
                #update("gweights", self.model.linear_layers[j].weight.grad, expand=True)
                #if not self.model.linear_layers[j].bias.requires_grad:
                    #update("gbiases", torch.zeros(self.model.linear_layers[j].bias.data.shape)) 
                #else:
                    #update("gbiases", self.model.linear_layers[j].bias.grad, expand=True)
                #update("gnet", self.model.gnet[j])
                #update("gact", self.model.gact[j])
                    
        #for layer_name in self.model.layer_names:
            #info[layer_name]["sgweights"] = np.sum(info[layer_name]["gweights"] , 0)
            #info[layer_name]["sgbiases"] = np.sum(info[layer_name]["gbiases"] , 0)

        if print_info:
            print("\n------------------------")
            self.print_dict(info)
        self.log.append(info)

    def print_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                print("\n{}".format(k))
                self.print_dict(v)
            else:
                print("\n{}".format(k))
                print(v)
                if isinstance(v, np.ndarray):
                    print(v.shape)
            
    def print_progress(self, epoch, loss):
        out = "Epoch {} training loss = ".format(epoch)
        out += "{0:.4f}".format(loss)
        print(out)
        
    def print_final(self):
        out = "Run {}:".format(self.name)
        out += " Loss at epoch {}:".format(self.loss_data["epochs"][0])
        out += " {0:.4f};".format(self.loss_data["train_losses"][0])
        out += " Accuracy:".format(self.loss_data["epochs"][0])
        out += " {0:.4f};".format(self.loss_data["train_accs"][0])
        out += " Last epoch: {};".format(self.loss_data["epochs"][-1])
        out += " Loss on last epoch: {0:.4f}".format(self.loss_data["train_losses"][-1])
        out += " Accuracy on last epoch: {0:.4f}".format(self.loss_data["train_accs"][-1])
        print(out)

    def plot_loss(self):
        fig, ax = plt.subplots(figsize=[12, 8])
        ax.plot(self.loss_data["epochs"], self.loss_data["train_losses"])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        max_loss = max(self.loss_data["train_losses"])
        ax.set_xlim([0, self.num_training_epochs])
        ax.set_ylim([0, max_loss + 0.1])
        ax.set_title('Training Progress: Loss')
        
    def plot_acc(self):
        fig, ax = plt.subplots(figsize=[12, 8])
        ax.plot(self.loss_data["epochs"], self.loss_data["train_accs"])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        max_loss = max(self.loss_data["train_accs"])
        ax.set_xlim([0, self.num_training_epochs])
        ax.set_ylim([0, max_loss + 0.1])
        ax.set_title('Training Progress: Accuracy')

    def train(self):
        if self.start_epoch == self.num_training_epochs - 1:
            # already trained
            return
        
        for epoch in range(self.start_epoch, self.num_training_epochs):
            # visualization purposes only
            if (epoch % self.test_freq) == 0:
                self.eval_test_points(epoch)
            
            # divide the training data into batches; 
            # we see each training item exactly once per epoch
            train_batches = self.dataset.get_per_epoch_batches(self.train_batch_size)
            
            loss_per_batch, acc_per_batch = [], []
            for batch in train_batches:
                # grab batch of training data
                inputs, targets = batch[0], batch[1]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get model predictions given inputs
                predictions = self.model.forward(inputs, record_data=False)
                
                # compute loss
                loss = self.criterion(predictions, targets)
                loss_per_batch.append(loss.data.numpy()) # record
                
                # compute accuracy (for visualization purposes)
                acc = torch.mean(1 - torch.abs(targets - torch.round(predictions)))
                acc_per_batch.append(acc.data.numpy())

                # update
                loss.backward()
                self.optimizer.step()
                
            # compute epoch loss, acc (for visualization purposes)
            epoch_loss = np.sum(loss_per_batch)
            epoch_acc = np.mean(acc_per_batch)
            
            # record
            self.loss_data["epochs"].append(epoch)
            self.loss_data["train_losses"].append(float(epoch_loss))
            self.loss_data["train_accs"].append(float(epoch_acc))
            
            if self.print_freq > 0:
                if (epoch % self.print_freq) == 0:
                    self.print_progress(epoch, epoch_loss)
                
            if (epoch % self.save_freq) == 0:
                self.save_results()
                
            if (epoch % self.checkpoint_freq) == 0:
                self.save_checkpoint(epoch)
            
            if epoch_loss <= self.stopping_criterion:
                self.final_save_and_view(epoch, epoch_loss)
                return
            
            if epoch == self.num_training_epochs - 1:
                self.final_save_and_view(epoch, epoch_loss)
                return
