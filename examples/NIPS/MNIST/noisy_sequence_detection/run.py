import time

import os
import re
import sys

import click

from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.MNIST.mnist import MNIST_Net, test_MNIST, MNISTNetFrozenEncoder
from examples.NIPS.MNIST.mnist import neural_predicate as default_neural_predicate

sys.path.append('../../../')
from train import train_model, train, batch_train_model, train_batch, epoch_train_model
from test_utils import get_confusion_matrix, calculate_f1
from data_loader import load
from model import Model, Var
from optimizer import Optimizer
from network import Network
import torch


def add_files_to(problog_files, problog_string):
    for problog_file in problog_files:
        with open(problog_file) as f:
            problog_string += f.read()
            problog_string += '\n\n'

    return problog_string


def my_test(model_to_test, test_queries, test_functions=None, to_file=None, confusion_index=None):
    res = test(
        model_to_test, test_queries, test_functions=test_functions, to_file=to_file, confusion_index=confusion_index
    )

    # res += test_MNIST(model_to_test)

    return res


def my_validation(model_to_validate, validation_queries):
    res = test(
        model_to_validate, validation_queries
    )

    return res[1][1]


def run(training_data, val_data, test_data, problog_files, problog_train_files=(), problog_val_files=(),
        problog_test_files=(), neural_predicate=default_neural_predicate, load_weights=None, snapshot_name=None,
        max_epochs=100, nn_model=MNIST_Net, nn_name='mnist_net', validation_function=my_validation,
        test_function=None):
    if snapshot_name is None:
        snapshot_name = training_data.split('/')[1]

    queries = load(training_data)
    val_queries = load(val_data)
    test_queries = load(test_data)

    problog_train_string, problog_val_string, problog_test_string = make_problog_strings(
        problog_files, problog_test_files, problog_train_files, problog_val_files
    )

    model_to_train, model_to_val, model_to_test, optimizer = make_train_val_test_models(
        neural_predicate, problog_train_string, problog_val_string, problog_test_string, training_caching=True,
        load_weights=load_weights, nn_model=nn_model, nn_name=nn_name
    )

    if max_epochs > 0:
        _, best_epoch, best_weights_fname = epoch_train_model(
            model_to_train,
            queries,
            max_epochs,
            optimizer,
            validation=lambda _: validation_function(
                model_to_val,
                val_queries
            ),
            # validation=None,
            log_epoch=1,
            snapshot_name='snapshots/model_{}'.format(snapshot_name),
            patience=10
        )
    elif load_weights:
        print("Testing loaded weights without training from: {}".format(load_weights))

        best_weights_fname = load_weights
    else:
        raise Exception("Max epochs smaller than 0 not allowed without loading weights")

    model_to_test.load_state(best_weights_fname)

    if test_function:
        test_function(model_to_test, test_queries)


def make_problog_strings(problog_files, problog_test_files, problog_train_files, problog_val_files):
    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_val_string = add_files_to(problog_val_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)

    return problog_train_string, problog_val_string, problog_test_string


def make_train_val_test_models(neural_predicate, problog_train_string, problog_val_string, problog_test_string,
                               training_caching=False, load_weights=None, nn_model=MNIST_Net, nn_name='mnist_net'):
    if nn_model is None:
        nn_model = MNIST_Net

    network = nn_model()
    net = Network(network, nn_name, neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model_to_train = Model(problog_train_string, [net], caching=training_caching)
    optimizer = Optimizer(model_to_train, 2)

    if load_weights:
        print("Loading weights from {}".format(load_weights))
        model_to_train.load_state(load_weights, strict=False)

    model_to_val = Model(problog_val_string, [net], caching=False)
    model_to_test = Model(problog_test_string, [net], caching=False)

    return model_to_train, model_to_val, model_to_test, optimizer


def get_problog_file_for(directory, folder, filename):
    prob_ec_cached = '{}/{}/{}'.format(directory, folder, filename)
    if not os.path.isfile(prob_ec_cached):
        prob_ec_cached = 'ProbLogFiles/{}'.format(filename)
    return prob_ec_cached


@click.command()
@click.option('--scenario', default='')
@click.option('--noise', default='')
@click.option('--directory', default='./scenarios100')
@click.option('--max_epochs', default=100)
@click.option('--mnist_classes', default=10)
@click.option('--load_weights', default=None)
@click.option('--freeze_encoder/--unfrozen_encoder', default=False)
def execute_scenarios(scenario, noise, directory, max_epochs, mnist_classes, load_weights, freeze_encoder):
    def test_function(*args1, **kwargs1):
        return my_test(
            *args1, **kwargs1, test_functions={
                'mnist_net': lambda *args2, **kwargs2: default_neural_predicate(
                    *args2, **kwargs2, dataset='test'
                )
            }
        )

    if freeze_encoder:
        nn_model = MNISTNetFrozenEncoder
    else:
        nn_model = MNIST_Net

    run_directory(
        directory=directory,
        scenario=scenario,
        noise=noise,
        load_weights=load_weights,
        nn_model=lambda: nn_model(N=mnist_classes),
        nn_name='mnist_net',
        max_epochs=max_epochs,
        validation_function=my_validation,
        test_function=test_function,
        neural_predicate=default_neural_predicate,
        class_type='digit',
    )


def run_directory(directory, scenario, noise, load_weights, nn_model, nn_name, max_epochs, validation_function,
                  test_function, neural_predicate, class_type='digit'):
    for folder in sorted(os.listdir(directory)):
        if folder.startswith('scenario') and re.search(scenario, folder):
            print("#######################################################################################")
            print(folder)

            run_folder(
                directory=directory,
                folder=folder,
                noise=noise,
                neural_predicate=neural_predicate,
                load_weights=load_weights,
                nn_model=nn_model,
                nn_name=nn_name,
                max_epochs=max_epochs,
                validation_function=validation_function,
                test_function=test_function,
                class_type=class_type
            )


def run_folder(directory, folder, noise, neural_predicate=default_neural_predicate, load_weights=None,
               nn_model=MNIST_Net, nn_name='mnist_net', max_epochs=100, validation_function=my_validation,
               test_function=None, class_type='digit'):
    prob_ec_cached = get_problog_file_for(directory, folder, 'prob_ec_cached.pl')
    event_defs = get_problog_file_for(directory, folder, 'event_defs.pl')

    for subfolder in sorted(os.listdir('{}/{}'.format(directory, folder))):
        if subfolder == '__pycache__':
            continue
        # if subfolder != 'noise_1_00':
        #     continue

        if re.search(noise, subfolder) and os.path.isdir('{}/{}/{}'.format(directory, folder, subfolder)):
            print('===================================================================================')
            print(subfolder)

            run_subfolder(
                directory, folder, subfolder, prob_ec_cached, event_defs, neural_predicate,
                load_weights=load_weights, nn_model=nn_model, nn_name=nn_name, max_epochs=max_epochs,
                validation_function=validation_function, test_function=test_function, class_type=class_type
            )


def run_subfolder(directory, folder, subfolder, prob_ec_cached, event_defs, neural_predicate=default_neural_predicate,
                  load_weights=None, nn_model=MNIST_Net, nn_name='mnist_net', max_epochs=100,
                  validation_function=my_validation, test_function=None, class_type='digit'):
    run(
        training_data='{}/{}/{}/init_train_data_clean.txt'.format(directory, folder, subfolder),
        # training_data='{}/{}/{}/digits_train_data.txt'.format(directory, folder, subfolder),
        # training_data='{}/{}/{}/sounds_train_data.txt'.format(directory, folder, subfolder),
        val_data='{}/{}/{}/init_val_data.txt'.format(directory, folder, subfolder),
        # val_data='{}/{}/{}/init_val_data_clean.txt'.format(directory, folder, subfolder),
        # val_data='{}/{}/{}/digits_val_data.txt'.format(directory, folder, subfolder),
        # val_data='{}/{}/{}/sounds_val_data.txt'.format(directory, folder, subfolder),
        # val_data='{}/{}/{}/sounds_train_data_small.txt'.format(directory, folder, subfolder),
        test_data='{}/{}/{}/init_{}_test_data.txt'.format(directory, folder, subfolder, class_type),
        # test_data='{}/{}/{}/init_test_data.txt'.format(directory, folder, subfolder),
        # test_data='{}/{}/init_train_data.txt'.format(folder, subfolder),
        # test_data='{}/{}/{}/sounds_train_data.txt'.format(directory, folder, subfolder),
        problog_files=[
            prob_ec_cached,
            event_defs
        ],
        problog_train_files=[
            '{}/{}/{}/in_train_data.txt'.format(directory, folder, subfolder)
        ],
        problog_val_files=[
            '{}/{}/{}/in_val_data.txt'.format(directory, folder, subfolder)
        ],
        problog_test_files=[
            '{}/{}/{}/in_test_data.txt'.format(directory, folder, subfolder)
            # '{}/{}/in_train_data.txt'.format(folder, subfolder)
        ],
        neural_predicate=neural_predicate,
        load_weights=load_weights,
        snapshot_name='{}_{}'.format(folder, subfolder),
        max_epochs=max_epochs,
        nn_model=nn_model,
        nn_name=nn_name,
        validation_function=validation_function,
        test_function=test_function
    )


if __name__ == '__main__':
    execute_scenarios()
