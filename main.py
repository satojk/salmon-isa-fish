import src.data as d_m
import src.model as m_m
import src.train as t_m

def main():    
    # PARAMETERS 
    # training params
    num_training_epochs = 1000 # 3000
    stopping_criterion = 0.0 # 0.0
    train_batch_size = 1 # 1
    
    # optimization params
    learning_rate = 0.1 # 0.01
    
    # model params
    representation_nonlinearity = "sigmoid" # "sigmoid"
    hidden_nonlinearity = "sigmoid" # "sigmoid"
    
    # initialization params
    weight_initialization_range = [-0.1, 0.1] # [-0.1, 0.1]
    output_layer_bias_val = -2. # -2.
    freeze_output_layer_biases = True # True
    
    # set up result dirs
    t_m.make_dir("logdirs")
    save_dir = t_m.get_logdir()
    print('Results will be saved in: {}'.format(save_dir))

    # create dataset
    data_fname = "data/data_01/data.csv"

    #new_items = [[],
                 #["tree", "flower", "fish", "bird"],
                 #["oak", "pine", "daisy", "rose", "sunfish", "salmon",	"canary", "robin"]]
    new_items = [[], [], []]
    representation_units_to_add = [0, 4, 3]

    for ix, items_to_consider in enumerate([
            ["plant", "animal"],
            ["plant", "animal", "tree", "flower", "bird", "fish"],
            None]):
        num_training_epochs = [1500, 2750, 4000][ix]
        dataset = d_m.Dataset(data_fname=data_fname, items_to_consider=items_to_consider)
        
        # create model
        # some model and train params are locked to dataset params
        item_input_size = len(dataset.item_names_to_inds)
        relation_input_size = len(dataset.relation_names_to_inds)
        representation_size = 7 + sum(representation_units_to_add[:ix])
        hidden_size = 15 # 15
        output_size = len(dataset.attribute_names_to_inds)
        
        model = m_m.Model(item_input_size=item_input_size,
                      relation_input_size=relation_input_size,
                      representation_size=representation_size,
                      hidden_size=hidden_size, 
                      output_size=output_size, 
                      representation_nonlinearity=representation_nonlinearity,
                      hidden_nonlinearity=hidden_nonlinearity,
                      wrange=weight_initialization_range, 
                      output_layer_bias_val=output_layer_bias_val,
                      freeze_output_layer_biases=freeze_output_layer_biases)

        # create trainer
        trainer = t_m.Trainer(dataset, model, 
                          name="{}".format(0),
                          train_batch_size=train_batch_size,
                          num_training_epochs=num_training_epochs,
                          stopping_criterion=stopping_criterion,
                          learning_rate=learning_rate, 
                          save_dir=save_dir,
                          test_freq=50,
                          print_freq=100,
                          show_plot=True,
                          new_items=new_items[ix],
                          representation_units_to_add=representation_units_to_add[ix])
        trainer.train()
    
if __name__ == '__main__':
    main()
