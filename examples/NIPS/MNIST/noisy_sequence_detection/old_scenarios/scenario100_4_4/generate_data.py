from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario001 import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario100_3_3.generate_data import \
    get_correct_digit_for_initiated_at

if __name__ == '__main__':
    generate_data(
        scenario_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=3),
        test_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=3),
        relevant_digits=3,
        noises_function=lambda: [0.0]
    )
