from examples.NIPS.generate_data_utils import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import \
    get_correct_digit_for_initiated_at


if __name__ == '__main__':
    generate_data(
        scenario_function=get_correct_digit_for_initiated_at,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=19,
        noises_function=lambda: [0.0],
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
