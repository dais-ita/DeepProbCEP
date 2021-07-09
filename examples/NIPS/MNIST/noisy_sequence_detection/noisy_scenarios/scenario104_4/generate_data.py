from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario001 import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario004.generate_data import get_random_assignment
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario103_2.generate_data import get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario104_2.generate_data import get_initiated_at_scenario104

if __name__ == '__main__':
    assignment = get_random_assignment(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario104(
            *args, **kwargs, assignment=assignment
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=3,
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
