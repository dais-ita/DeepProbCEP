import argparse

from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario103_2.generate_data import get_initiated_at_scenario103
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values, get_urban_sound_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute the strawman approach.')
    parser.add_argument('test_fold', metavar='N', type=int, help='the fold to use for testing')

    args = parser.parse_args()

    t_loader, v_loader = get_urban_sound_datasets(base_folder='../..', test_fold=args.test_fold)

    generate_data(
        scenario_function=get_initiated_at_scenario103,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=4,
        training_set=t_loader,
        testing_set=v_loader,
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        get_true_values=sound_true_values,
        network_clause='sound',
    )
