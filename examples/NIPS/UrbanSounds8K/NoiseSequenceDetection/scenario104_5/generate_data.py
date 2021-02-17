import argparse

from examples.NIPS.MNIST.noisy_sequence_detection.old_scenarios.scenario004.generate_data import get_random_assignment
from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.scenario104_2.generate_data import get_initiated_at_scenario104
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values, get_urban_sound_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute the strawman approach.')
    parser.add_argument('test_fold', metavar='N', type=int, help='the fold to use for testing')

    args = parser.parse_args()

    t_loader, v_loader = get_urban_sound_datasets(base_folder='../..', test_fold=args.test_fold)

    all_events = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]

    start_sequence = ['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren']
    end_sequence = ['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music']

    numbers_assignment = get_random_assignment(list(range(10)))

    assignment = list(map(lambda x: all_events[x], numbers_assignment))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario104(
            *args, **kwargs, assignment=assignment
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=4,
        training_set=t_loader,
        testing_set=v_loader,
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        get_true_values=sound_true_values,
        network_clause='sound',
    )
