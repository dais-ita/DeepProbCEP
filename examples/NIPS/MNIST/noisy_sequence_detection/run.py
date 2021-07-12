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


def run(training_data, test_data, problog_files, problog_train_files=(), problog_test_files=(), confusion_index=None,
        neural_predicate=default_neural_predicate, snapshots=True):
    start = time.time()

    queries = load(training_data)
    test_queries = load(test_data)

    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)
    print('Problog files prepared at {}'.format(time.time() - start))

    network = MNIST_Net()
    print('MNIST_Net prepared at {}'.format(time.time() - start))
    net = Network(network, 'mnist_net', neural_predicate)
    print('Network prepared at {}'.format(time.time() - start))
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    print('net.optimizer prepared at {}'.format(time.time() - start))
    model_to_train = Model(problog_train_string, [net], caching=True)
    print('Model to train prepared at {}'.format(time.time() - start))
    optimizer = Optimizer(model_to_train, 2)
    print('Optimizer prepared at {}'.format(time.time() - start))

    model_to_test = Model(problog_test_string, [net], caching=False)
    print('Model to test prepared at {}'.format(time.time() - start))

    train_model(
        model_to_train,
        queries,
        2,
        optimizer,
        test_iter=len(queries) * 2,
        # test_iter=100,
        test=lambda _: my_test(
            model_to_test,
            test_queries,
            test_functions={
                'mnist_net': lambda *args, **kwargs: neural_predicate(
                    *args, **kwargs, dataset='test'
                )
            },
            to_file='to_file.txt',
            confusion_index=confusion_index
        ),
        log_iter=len(queries),
        snapshot_iter=len(queries) * 5 if snapshots else None,
        snapshot_name='snapshots/model'
    )

    # my_test(
    #     model_to_test, test_queries, test_functions={
    #         'mnist_net': lambda *args, **kwargs: neural_predicate(
    #             *args, **kwargs, dataset='test'
    #         )
    #     }
    # )


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
                training_data='{}/{}/{}/init_train_data_clean_500.txt'.format(directory, folder, subfolder),
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
