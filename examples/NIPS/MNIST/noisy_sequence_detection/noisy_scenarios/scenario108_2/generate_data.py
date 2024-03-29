import random

from examples.NIPS.generate_data_utils import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario103_2.generate_data import get_correct_digit_for_initiated_at


def get_initiated_at_scenario108(digit, last_digits, threshold, available_digits, swap):
    if digit in last_digits:
        if digit == swap[0] and random.random() < threshold:
            return swap[1], False
        else:
            return digit, True

    return None, None


def make_scenario108_for_window(window):
    # available_digits = list(range(10))
    #
    # swap_from = random.choice(available_digits)
    # swap_to = random.choice(available_digits)
    # # Ensure that we are not swapping a number for itself
    # while swap_from == swap_to:
    #     swap_to = random.choice(available_digits)

    swap_from = 0
    swap_to = 1

    with open('swap.txt', 'w') as o:
        o.write('{},{}'.format(swap_from, swap_to))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario108(
            *args, **kwargs, swap=(swap_from, swap_to)
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=window,
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )


if __name__ == '__main__':
    make_scenario108_for_window(1)
