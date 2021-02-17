from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario001 import generate_data


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit in last_digits:
        return digit, True

    return None, None


if __name__ == '__main__':
    generate_data(
        scenario_function=get_correct_digit_for_initiated_at,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=9,
        noises_function=lambda: [0.0],
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
