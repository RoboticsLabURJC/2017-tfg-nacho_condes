from network import network
import os.path
import time

myNet = network.Network()


def train(final_test=False):
    ''' This function trains and saves the CNN myNet, by first asking for the parameters to the user'''
    start = time.time()
    # Parameter acquisition.
    model_path = raw_input('Enter the path to save the network (leave blank for "my-model"): ')
    if model_path == '':
        model_path = 'my-model'

    training_dataset_path = raw_input('Enter the path to the training dataset .h5 file (leave blank to use standard MNIST): ')
    while training_dataset_path != '' and not os.path.isfile(training_dataset_path):
        training_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')

    validation_dataset_path = raw_input('Enter the path to the validation dataset .h5 file (leave blank to use standard MNIST): ')
    while validation_dataset_path != '' and not os.path.isfile(validation_dataset_path):
        validation_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')
    
    train_steps = raw_input('Enter train steps (leave blank to 2000): ')
    if train_steps == '':
        train_steps = 2000
    else:
        train_steps = int(train_steps)

    batch_size = raw_input('Enter batch size (leave blank to 50): ')
    if batch_size == '':
        batch_size = 50
    else:
        batch_size = int(batch_size)

    early_stopping = raw_input('Do you want to implement early stopping (y/n)? ')
    while early_stopping != 'y' and early_stopping != 'n':
        early_stopping = raw_input('    Please enter y (yes) or n (no): ')

    monitor = None
    patience =  None
    if early_stopping == 'y':
        monitor = raw_input('    What do you want to monitor (accuracy/loss)? ')
        while monitor != 'accuracy' and monitor != 'loss':
            monitor = raw_input('    Please enter "accuracy" or "loss": ')
        patience = raw_input('    Enter patience (leave blank for 5): ')
        if patience == '':
            patience = 5
        else:
            patience = int(patience)

    # Training API call
    parameters = {'training_dataset_path': training_dataset_path,
                  'validation_dataset_path': validation_dataset_path,
                  'learning_rate': 1e-4,
                  'train_steps': train_steps,
                  'batch_size': batch_size,
                  'model_path': model_path,
                  'early_stopping': early_stopping,
                  'monitor': monitor,
                  'patience': patience
                  }
    (train_acc, train_loss, val_acc, val_loss) = myNet.train(parameters)

    elapsed = time.time() - start
    print("Elapsed training time: %.3f seconds." % elapsed)


    if final_test:
        test_parameters = get_test_parameters(is_training=True)
        test_parameters['train_acc'] = train_acc
        test_parameters['train_loss'] = train_loss
        test_parameters['val_acc'] = val_acc
        test_parameters['val_loss'] = val_loss
        test(test_parameters)



def get_test_parameters(is_training=False):
    ''' This function retrieves parameters for a CNN test'''

    # Parameter acquisition.
    test_dataset_path = raw_input('Enter the path to the testing dataset .h5 file (leave blank to use standard MNIST): ')
    while test_dataset_path != '' and not os.path.isfile(test_dataset_path):
        test_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')

    n_samples = raw_input('Enter the number of testing samples (leave blank to 2000): ')
    if n_samples == '':
        n_samples = 2000
    else:
        n_samples = int(n_samples)

    output_matrix = raw_input('Enter the desired name for the output matrix .mat file (leave blank to "results"): ')
    if output_matrix == '':
        output_matrix = 'results'
    
    output_matrix = (output_matrix + '.mat')


    test_parameters = {'test_dataset_path': test_dataset_path,
                       'n_samples': n_samples,
                       'is_training': is_training,
                       'output_matrix': output_matrix}
    return test_parameters



def test(test_parameters):
    ''' This function evaluates the CNN myNet, with previously fetched test parameters'''
    myNet.test(test_parameters)





action = None
while action != 'train' and action != 'test' and action != 'both':
    action = raw_input('\nWhat do you want to do (train/test/both)? ')


if action == 'train':
    train()

elif action == 'test':
    model_path = raw_input('Enter the path containing the model to evaluate (leave blank for "my-model"): ')
    while model_path != '' and not os.path.exists(model_path):
        model_path = raw_input('    Please enter a valid path (leave blank for "my-model")')
    if model_path == '':
        model_path = 'my-model'

    myNet.load(model_path)

    test_parameters = get_test_parameters()
    test(test_parameters)

else:
    train(final_test=True)