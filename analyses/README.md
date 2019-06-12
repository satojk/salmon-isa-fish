Results is structured as required for the FFBP visualizaion software:

results = {"loss_data": # training loss and accuracy data

                {"epochs": [], # epochs at which train loss, acc were recorded
                 "train_losses": [], # epoch loss
                 "train_accs": [] # epoch acc
                 }, 
                 
           "test_data": # data collected during evaluation of test set (inside `eval_test_points`) as list of dicts, one per evaluation event (e.g. given 3000 epochs and test_freq == 200, this will be 16 long
           
                [{"enum": int, # epoch on which the test occurred
                  "input": test_size x num_items + num_relations np array, # concatenation of the item and relation inputs
                  "target": test_size x num_attributes np array, 
                  "labels": test_size-long list of input descriptions as strings, # e.g. 'pine isa'
                  "item_labels": test_size-long list of item inputs as strings, # e.g. 'pine'
                  "relation_labels": test_size-long list of relation inputs as strings, # e.g. 'isa'
                  "loss_sum": float, # summed loss across test items
                  "loss": test_size-long list, # losses by item
                  "representation_layer": # info recorded for first model layer
                  
                      {"input_": test_size x layer_inp_sz np array, # layer input, as recorded in self.model.layer_inputs
                       "weights": np array,
                       "biases": np array,
                       "net": test_size x layer_output_sz np array, # output of linear layer, before nonlinearity, as recorded in self.model.layer_outputs
                       "act": test_size x layer_output_sz np array, # layer activations, as recorded in self.model.layer_activations
                       "gweights": test_size x weight_sz[0] x weight_sz[1] np array, # dE/dW 
                       "gbiases": test_size x bias_sz[0] np array, # dE/dB 
                       "gnet": test_size x layer_output_sz np array, #dE/dnet
                       "gact": test_size x layer_output_sz np array, #dE/da
                       "sgweights": weight_sz[0] x weight_sz[1] np array, # gweights summed across test items
                       "sgbiases": bias_sz-long np array # gbiases summed across test items
                       }
                       
                  "hidden_layer": same structure as representation_layer,
                  "output_layer": same structure as representation_layer,
                  }, ...]
            }
