import time

import os
import re
import sys

import click

from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.MNIST.mnist import MNIST_Net, test_MNIST
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
        problog_test_files=(), confusion_index=None, neural_predicate=default_neural_predicate, snapshots=True):
    start = time.time()

    scenario = training_data.split('/')[1]

    queries = load(training_data)
    val_queries = load(val_data)
    test_queries = load(test_data)

    problog_train_string, problog_val_string, problog_test_string = make_problog_strings(
        problog_files, problog_test_files, problog_train_files, problog_val_files
    )
    print('Problog files prepared at {}'.format(time.time() - start))

    model_to_train, model_to_val, model_to_test, optimizer = make_train_val_test_models(
        neural_predicate, problog_train_string, problog_val_string, problog_test_string, training_caching=True
    )

    _, best_epoch, best_weights_fname = epoch_train_model(
        model_to_train,
        queries,
        100,
        optimizer,
        validation=lambda _: my_validation(
            model_to_val,
            val_queries
        ),
        # validation=None,
        log_epoch=1,
        snapshot_name='snapshots/model_{}'.format(scenario),
        patience=10
    )

    model_to_test.load_state(best_weights_fname)

    my_test(
        model_to_test, test_queries, test_functions={
            'mnist_net': lambda *args, **kwargs: neural_predicate(
                *args, **kwargs, dataset='test'
            )
        }
    )


def make_problog_strings(problog_files, problog_test_files, problog_train_files, problog_val_files):
    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_val_string = add_files_to(problog_val_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)

    return problog_train_string, problog_val_string, problog_test_string


def make_train_val_test_models(neural_predicate, problog_train_string, problog_val_string, problog_test_string,
                               training_caching=False):
    network = MNIST_Net()
    net = Network(network, 'mnist_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model_to_train = Model(problog_train_string, [net], caching=training_caching)
    optimizer = Optimizer(model_to_train, 2)

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
@click.option('--snapshots/--no-snapshots', default=True)
def execute_scenarios(scenario, noise, directory, snapshots):
    for folder in sorted(os.listdir(directory)):
        if folder.startswith('scenario') and re.search(scenario, folder):
            print("#######################################################################################")
            print(folder)

            run_folder(directory, folder, noise, snapshots)


def run_folder(directory, folder, noise, snapshots):
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

            run(
                training_data='{}/{}/{}/init_train_data_clean.txt'.format(directory, folder, subfolder),
                # training_data='{}/{}/{}/digits_train_data.txt'.format(directory, folder, subfolder),
                val_data='{}/{}/{}/init_val_data.txt'.format(directory, folder, subfolder),
                # val_data='{}/{}/{}/init_val_data_clean.txt'.format(directory, folder, subfolder),
                # val_data='{}/{}/{}/digits_val_data.txt'.format(directory, folder, subfolder),
                test_data='{}/{}/{}/init_digit_test_data.txt'.format(directory, folder, subfolder),
                # test_data='{}/{}/{}/init_test_data.txt'.format(directory, folder, subfolder),
                # test_data='{}/{}/init_train_data.txt'.format(folder, subfolder),
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
                snapshots=snapshots
            )


if __name__ == '__main__':
    execute_scenarios()
