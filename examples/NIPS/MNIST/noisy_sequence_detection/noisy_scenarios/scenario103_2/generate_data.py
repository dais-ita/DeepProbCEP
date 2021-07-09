import random

from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario001 import generate_data


def get_initiated_at_scenario103(digit, last_digits, threshold, available_digits):
    if digit in last_digits:
        if random.random() < threshold:
            return random.choice(available_digits), False
        else:
            return digit, True

    return None, None


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit in last_digits:
        return digit, True

    return None, None


if __name__ == '__main__':
    generate_data(
        scenario_function=get_initiated_at_scenario103,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1,
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
