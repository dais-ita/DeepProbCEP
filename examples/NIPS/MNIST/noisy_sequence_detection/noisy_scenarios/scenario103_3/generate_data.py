from examples.NIPS.generate_data_utils import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario103_2.generate_data import get_initiated_at_scenario103, \
    get_correct_digit_for_initiated_at

if __name__ == '__main__':
    generate_data(
        scenario_function=get_initiated_at_scenario103,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=2,
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
