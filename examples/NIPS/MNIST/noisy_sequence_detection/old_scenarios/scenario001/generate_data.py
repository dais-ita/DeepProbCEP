import os

import torch
from torch.utils.data import random_split
import torchvision
from examples.NIPS.generate_data_utils import mnist_true_values, gather_examples, get_digit_for_initiated_at, \
    get_correct_digit_for_initiated_at, default_noises, default_folder_name, list_true_values


trainset = torchvision.datasets.MNIST(root='~/datasets/Image/MNIST', train=True, download=True)
indexed_trainset = mnist_true_values(trainset)

torch.manual_seed(42)  # Set the seed for reproducibility
indexed_trainset, indexed_valset = random_split(indexed_trainset, [50000, 10000])

testset = torchvision.datasets.MNIST(root='~/datasets/Image/MNIST', train=False, download=True)
indexed_testset = mnist_true_values(testset)


def generate_data(noises_function=default_noises, folder_name=default_folder_name,
                  scenario_function=get_digit_for_initiated_at, test_function=get_correct_digit_for_initiated_at,
                  relevant_digits=1, training_set=indexed_trainset, validating_set=indexed_valset,
                  testing_set=indexed_testset, start_sequence=None, end_sequence=None,
                  get_true_values=list_true_values, network_clause='digit', create_initiated_at=None, in_sequence=None,
                  create_initiated_wildcards_training=False, create_initiated_wildcards_testing=False):
    in_train_data = 'in_train_data.txt'
    init_train_data = 'init_train_data.txt'
    holds_train_data = 'holds_train_data.txt'
    digits_train_data = '{}s_train_data.txt'.format(network_clause)
    init_digit_train_data = 'init_{}_train_data.txt'.format(network_clause)
    in_val_data = 'in_val_data.txt'
    init_val_data = 'init_val_data.txt'
    holds_val_data = 'holds_val_data.txt'
    digits_val_data = '{}s_val_data.txt'.format(network_clause)
    init_digit_val_data = 'init_{}_val_data.txt'.format(network_clause)
    in_test_data = 'in_test_data.txt'
    init_test_data = 'init_test_data.txt'
    holds_test_data = 'holds_test_data.txt'
    digits_test_data = '{}s_test_data.txt'.format(network_clause)
    init_digit_test_data = 'init_{}_test_data.txt'.format(network_clause)

    for noise in noises_function():
        folder = folder_name(noise)

        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        iteration_in_train_data = folder + in_train_data
        iteration_init_train_data = folder + init_train_data
        iteration_holds_train_data = folder + holds_train_data
        iteration_digits_train_data = folder + digits_train_data
        iteration_init_digit_train_data = folder + init_digit_train_data
        iteration_in_val_data = folder + in_val_data
        iteration_init_val_data = folder + init_val_data
        iteration_holds_val_data = folder + holds_val_data
        iteration_digits_val_data = folder + digits_val_data
        iteration_init_digit_val_data = folder + init_digit_val_data
        iteration_in_test_data = folder + in_test_data
        iteration_init_test_data = folder + init_test_data
        iteration_holds_test_data = folder + holds_test_data
        iteration_digits_test_data = folder + digits_test_data
        iteration_init_digit_test_data = folder + init_digit_test_data

        gather_examples(
            dataset=training_set,
            in_filename=iteration_in_train_data,
            initiated_filename=iteration_init_train_data,
            holds_filename=iteration_holds_train_data,
            network_filename=iteration_digits_train_data,
            init_network_filename=iteration_init_digit_train_data,
            threshold=noise,
            scenario_function=scenario_function,
            relevant_digits=relevant_digits,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            get_true_values=get_true_values,
            network_clause=network_clause,
            create_initiated_at=create_initiated_at,
            in_sequence=in_sequence,
            create_initiated_wildcards=create_initiated_wildcards_training
        )
        gather_examples(
            dataset=validating_set,
            in_filename=iteration_in_val_data,
            initiated_filename=iteration_init_val_data,
            holds_filename=iteration_holds_val_data,
            network_filename=iteration_digits_val_data,
            init_network_filename=iteration_init_digit_val_data,
            threshold=0.0,
            scenario_function=test_function,
            relevant_digits=relevant_digits,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            get_true_values=get_true_values,
            network_clause=network_clause,
            create_initiated_at=create_initiated_at,
            in_sequence=in_sequence,
            create_initiated_wildcards=create_initiated_wildcards_testing
        )
        gather_examples(
            dataset=testing_set,
            in_filename=iteration_in_test_data,
            initiated_filename=iteration_init_test_data,
            holds_filename=iteration_holds_test_data,
            network_filename=iteration_digits_test_data,
            init_network_filename=iteration_init_digit_test_data,
            threshold=0.0,
            scenario_function=test_function,
            relevant_digits=relevant_digits,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            get_true_values=get_true_values,
            network_clause=network_clause,
            create_initiated_at=create_initiated_at,
            in_sequence=in_sequence,
            create_initiated_wildcards=create_initiated_wildcards_testing
        )


if __name__ == '__main__':
    generate_data()
