import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, item_input_size, 
                 relation_input_size, 
                 representation_size,
                 hidden_size, 
                 output_size, 
                 representation_nonlinearity="sigmoid",
                 hidden_nonlinearity="sigmoid",
                 final_nonlinearity="sigmoid",
                 wrange=[-0.9, 0.9],
                 output_layer_bias_val=-2.,
                 freeze_output_layer_biases=True):
        super(Model, self).__init__()
        self.item_input_size = item_input_size
        self.relation_input_size = relation_input_size
        self.representation_size = representation_size
        self.hidden_size = hidden_size
        self.wrange = wrange
        # specify network layers
        self.linear_layers = nn.ModuleList(
                                [nn.Linear(item_input_size, representation_size),
                                 nn.Linear(representation_size + relation_input_size, 
                                           hidden_size),
                                 nn.Linear(hidden_size, output_size)])
        self.nonlinearities = [self.select_nonlinearity(hidden_nonlinearity), 
                               self.select_nonlinearity(representation_nonlinearity),
                               self.select_nonlinearity(final_nonlinearity)] 
        self.layer_names = ["representation_layer", "hidden_layer", "output_layer"]
    
        # initialize weights
        for layer, layer_name in zip(self.linear_layers, self.layer_names):
            self.init_weights(layer, layer_name, wrange, 
                              output_layer_bias_val, 
                              freeze_output_layer_biases=freeze_output_layer_biases)
            
    def select_nonlinearity(self, nonlinearity):
        if nonlinearity == "sigmoid":
            return nn.Sigmoid()
        elif nonlinearity == "tanh":
            return nn.Tanh()
        elif nonlinearity == "relu":
            return nn.ReLU()
        elif nonlinearity == "none":
            return lambda x: x
        
    def init_weights(self, layer, layer_name, wrange, output_layer_bias_val, 
                     freeze_output_layer_biases=True):
        layer.weight.data.uniform_(wrange[0], wrange[1]) # inplace
        if layer_name == "output_layer":
            if not output_layer_bias_val is None:
                layer.bias.data = output_layer_bias_val * torch.ones(layer.bias.data.shape)
            else:
                layer.bias.data.uniform_(wrange[0], wrange[1])
            if freeze_output_layer_biases:
                layer.bias.requires_grad = False
        else:
            layer.bias.data.uniform_(wrange[0], wrange[1])

    def add_representation_units(self, n):
        print("Going from {} to {} representation units".format(self.representation_size, self.representation_size + n))
        previous_weights = [self.linear_layers[0].weight.data,
                            self.linear_layers[1].weight.data]
        new_weights_in = torch.FloatTensor(n, self.item_input_size).uniform_(self.wrange[0], self.wrange[1])
        new_weights_out = torch.FloatTensor(self.hidden_size, n).uniform_(self.wrange[0], self.wrange[1])
        new_biases = torch.FloatTensor(n).uniform_(self.wrange[0], self.wrange[1])

        new_weights_in = torch.cat((new_weights_in, previous_weights[0]), dim=0)
        new_weights_out = torch.cat((new_weights_out, previous_weights[1]), dim=1)
        new_biases = torch.cat((new_biases, self.linear_layers[0].bias.data), dim=0)

        self.linear_layers[0] = nn.Linear(self.item_input_size, self.representation_size + n)
        self.linear_layers[1] = nn.Linear(self.representation_size + self.relation_input_size + n, self.hidden_size)

        with torch.no_grad():
            self.linear_layers[0].weight = nn.Parameter(new_weights_in)
            self.linear_layers[1].weight = nn.Parameter(new_weights_out)
            self.linear_layers[0].bias = nn.Parameter(new_biases)


    def copy_weights_to_rep(self, unit_from, unit_to):
        with torch.no_grad():
            for ix, receiving_unit in enumerate(self.linear_layers[0].weight):
                self.linear_layers[0].weight[ix][unit_to] = self.linear_layers[0].weight[ix][unit_from]

            
    def print_model_layers(self):
        print("\nModel linear layers:")
        print(self.linear_layers)
        print("\nModel nonlinearities:")
        print(self.nonlinearities)
        print("")

    def record_gnet(self, grad):
        self.gnet.append(grad)
        
    def record_gact(self, grad):
        self.gact.append(grad)

    def forward(self, inp, record_data=True):
        if record_data:
            # for visualization purposes; not necessary to train the model 
            self.layer_inputs, self.layer_outputs, self.layer_activations = [], [], []
            self.gnet, self.gact = [], [] # these are recorded in backward order (the order of the gradient computations)
        
        for i, (layer, nonlinearity) in enumerate(zip(self.linear_layers, 
                                                      self.nonlinearities)):
            
            if i == 0:
                out = inp["item"] # input to first layer is just the item
            elif i == 1:
                out = torch.cat((out, inp["relation"]), 1) # input to second layer is cat(representation, relation)
                
            if record_data: # visualization purposes only
                self.layer_inputs.append(out)
                
            out = layer(out)
            
            if record_data:
                self.layer_outputs.append(out)
                out.register_hook(self.record_gnet) # dE/dnet
                
            # apply nonlinearity
            out = nonlinearity(out)
            
            if record_data:
                self.layer_activations.append(out)
                out.register_hook(self.record_gact) # dE/da
        return out           
