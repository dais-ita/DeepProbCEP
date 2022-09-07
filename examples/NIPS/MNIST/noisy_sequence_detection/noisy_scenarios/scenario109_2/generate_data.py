import random

from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario001 import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario103_2.generate_data import get_correct_digit_for_initiated_at


def get_initiated_at_scenario109(digit, last_digits, threshold, available_digits, target):
    if digit in last_digits:
        if digit == target and random.random() < threshold:
            return random.choice(available_digits), False
        else:
            return digit, True

    return None, None


if __name__ == '__main__':
    targeted_digit = 0

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario109(
            *args, **kwargs, target=targeted_digit
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1,
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
