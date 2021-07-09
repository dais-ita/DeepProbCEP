import argparse

from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values, get_urban_sound_datasets, \
    get_urban_sound_datasets_with_validation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute the strawman approach.')
    parser.add_argument('val_fold', metavar='N', type=int, help='the fold to use for validation')
    parser.add_argument('test_fold', metavar='N', type=int, help='the fold to use for testing')

    args = parser.parse_args()

    training, validation, testing = get_urban_sound_datasets_with_validation(
        validation_fold=args.val_fold, test_fold=args.test_fold
    )

    generate_data(
        noises_function=lambda: [0.0],
        scenario_function=get_correct_digit_for_initiated_at,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=2,
        training_set=training,
        validating_set=validation,
        testing_set=testing,
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        get_true_values=sound_true_values,
        network_clause='sound',
        create_initiated_wildcards_training=True,
        create_initiated_wildcards_testing=True
    )
