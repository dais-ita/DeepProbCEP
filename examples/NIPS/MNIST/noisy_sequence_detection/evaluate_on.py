import time

import os
import re
import sys

import click

from examples.NIPS.MNIST.mnist import MNIST_Net
from examples.NIPS.MNIST.mnist import neural_predicate as default_neural_predicate
from examples.NIPS.MNIST.noisy_sequence_detection.run import my_test, get_problog_file_for, add_files_to, \
    make_train_val_test_models, make_problog_strings, run

sys.path.append('../../../')
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
import torch


def evaluate_on(load_from, test_data, problog_files, problog_test_files=(), neural_predicate=default_neural_predicate):
    test_queries = load(test_data)

    problog_string = add_files_to(problog_files, '')

    problog_test_string = add_files_to(problog_test_files, problog_string)

    network = MNIST_Net()
    net = Network(network, 'mnist_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    model_to_test = Model(problog_test_string, [net], caching=False)

    print(load_from)
    model_to_test.load_state(load_from)

    my_test(
        model_to_test, test_queries, test_functions={
            'mnist_net': lambda *args, **kwargs: neural_predicate(
                *args, **kwargs, dataset='test'
            )
        }
    )


@click.command()
@click.option('--scenario', default='')
@click.option('--noise', default='')
@click.option('--directory', default='./scenarios100')
@click.option('--load_from', default='snapshots/model_scenario100_5_5000_epoch_0035.mdl')
def evaluate_scenarios(scenario, noise, directory, load_from):
    for folder in sorted(os.listdir(directory)):
        if folder.startswith('scenario') and re.search(scenario, folder):
            print("#######################################################################################")
            print(folder)

            evaluate_folder(directory, folder, noise, load_from)


def evaluate_folder(directory, folder, noise, load_from):
    prob_ec_cached = get_problog_file_for(directory, folder, 'prob_ec_cached.pl')
    event_defs = get_problog_file_for(directory, folder, 'event_defs.pl')

    for subfolder in sorted(os.listdir('{}/{}'.format(directory, folder))):
        if subfolder == '__pycache__':
            continue

        if re.search(noise, subfolder) and os.path.isdir('{}/{}/{}'.format(directory, folder, subfolder)):
            print('===================================================================================')
            print(subfolder)

            evaluate_on(
                load_from=load_from,
                test_data='{}/{}/{}/init_digit_test_data.txt'.format(directory, folder, subfolder),
                problog_files=[
                    prob_ec_cached,
                    event_defs
                ],
                problog_test_files=[
                    '{}/{}/{}/in_test_data.txt'.format(directory, folder, subfolder)
                ]
            )


if __name__ == '__main__':
    evaluate_scenarios()
