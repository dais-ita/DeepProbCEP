import random

import numpy as np
import os

import torch
from torch.utils.data import random_split
import torchvision


def write_initiated_ats(init_f, init_digit_f, clause, sequence, value, timestamp):
    initiated_at = '{}({} = {}, {}).\n'.format(clause, sequence, value, timestamp)

    init_f.write(initiated_at)
    init_digit_f.write(initiated_at)


def default_create_initiated_at(digit, start_sequence, end_sequence, init_f, init_digit_f, in_sequence, t, is_true):
    if digit is None:
        write_initiated_ats(
            init_f,
            init_digit_f,
            'initiatedAt',
            'X',
            'Y',
            t
        )
    elif digit in start_sequence:
        seq_id = start_sequence.index(digit)

        seq = 'sequence{}'.format(seq_id)

        write_initiated_ats(
            init_f,
            init_digit_f,
            'initiatedAt{}'.format('' if is_true else 'Noise'),
            seq,
            'true',
            t
        )

        in_sequence[seq] = True
    elif digit in end_sequence:
        seq_id = end_sequence.index(digit)

        seq = 'sequence{}'.format(seq_id)

        write_initiated_ats(
            init_f,
            init_digit_f,
            'initiatedAt{}'.format('' if is_true else 'Noise'),
            seq,
            'false',
            t
        )

        in_sequence[seq] = False


def mnist_true_values(dataset):
    return [
        (ident, digit)
        for ident, (image, digit) in enumerate(dataset)
    ]


def list_true_values(dataset):
    return list(dataset)


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, network_filename, init_network_filename,
                    threshold, scenario_function, relevant_digits, start_sequence=None, end_sequence=None,
                    get_true_values=mnist_true_values, network_clause='digit', create_initiated_at=None,
                    in_sequence=None, create_initiated_wildcards=False):
    if start_sequence is None:
        start_sequence = [0, 2, 4, 6, 8]
    if end_sequence is None:
        end_sequence = [1, 3, 5, 7, 9]

    if create_initiated_at is None:
        create_initiated_at = default_create_initiated_at

    if in_sequence is None:
        in_sequence = {
            'sequence{}'.format(i): False
            for i in range(5)
        }

    available_digits = start_sequence + end_sequence

    true_values = get_true_values(dataset)

    random.shuffle(true_values)

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(network_filename, 'w') as network_f:
                    with open(init_network_filename, 'w') as init_network_f:
                        last_network = []

                        for t, (image, network) in enumerate(true_values):
                            network_f.write('{}({}, {}).\n'.format(network_clause, image, network))
                            init_network_f.write('{}({}, {}).\n'.format(network_clause, image, network))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            for seq, val in in_sequence.items():
                                holds_f.write(
                                    'holdsAt({} = {}, {}).\n'.format(
                                        seq, str(val).lower(), t
                                    )
                                )

                            if last_network:
                                digit_to_create, is_true = scenario_function(
                                    network, last_network, threshold, available_digits
                                )

                                if digit_to_create is not None or create_initiated_wildcards:
                                    create_initiated_at(
                                        digit_to_create, start_sequence, end_sequence,
                                        init_f, init_network_f, in_sequence, t, is_true
                                    )

                            last_network.append(network)
                            last_network = last_network[-relevant_digits:]

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, range(len(true_values))))
                            )
                        )


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit == last_digits[-1]:
        return digit, True
    elif random.random() < threshold:
        return digit, False

    return None, None


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit == last_digits[-1]:
        return digit, True

    return None, None


def default_noises(min_noise=0.0, max_noise=1.1, noise_step=0.2):
    return np.arange(min_noise, max_noise, noise_step)


def default_folder_name(noise):
    return 'noise_{0:.2f}/'.format(noise).replace('.', '_')


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


def check_swap_for_itself(assignment):
    for i, a in enumerate(assignment):
        if i == a:
            return True

    return False


def get_random_assignment(digits):
    assignment = digits

    while check_swap_for_itself(assignment):
        random.shuffle(assignment)

    return assignment
