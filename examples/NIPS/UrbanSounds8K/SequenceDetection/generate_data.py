import json
import argparse

import torchvision
from examples.NIPS.generate_data_utils import gather_examples
import pandas as pd


preprocessable = pd.read_pickle(
    '/home/roigvilamalam/projects/deepproblog/examples/NIPS/UrbanSounds8K/preprocessable.pkl'
)
preprocessable = preprocessable[preprocessable['preprocessable']]

preprocessable_filenames = set(preprocessable['filename'])


# def sound_true_values(dataset):
#     return [
#         (
#             "'examples/NIPS/UrbanSounds8K{}'".format(sample['path'][2:]),
#             sample['class']
#         )
#         for sample in dataset.dataset.data_arr
#         if sample['path'][3:] in preprocessable_filenames
#     ]


def sound_true_values(dataset):
    return [
        (
            "'~/datasets/audio/UrbanSounds8K/audio/fold{}/{}'".format(sample['fold'], sample['slice_file_name']),
            sample['class']
        )
        for i, sample in dataset.iterrows()
    ]


# def get_urban_sound_datasets(base_folder='..'):
#     config = json.load(open('{}/my-config_generate.json'.format(base_folder)))
#
#     data_manager = getattr(data_module, config['data']['type'])(config['data'])
#
#     t_loader = data_manager.get_loader('train', transfs=None)
#     v_loader = data_manager.get_loader('val', transfs=None)
#
#     return t_loader, v_loader


def get_urban_sound_datasets(base_folder='..', test_fold=10):
    t_loader = preprocessable[preprocessable['fold'] != test_fold]
    v_loader = preprocessable[preprocessable['fold'] == test_fold]

    return t_loader, v_loader


def get_urban_sound_datasets_with_validation(validation_fold=9, test_fold=10):
    training = preprocessable[(preprocessable['fold'] != validation_fold) & (preprocessable['fold'] != test_fold)]
    validation = preprocessable[preprocessable['fold'] == validation_fold]
    testing = preprocessable[preprocessable['fold'] == test_fold]

    return training, validation, testing


def get_urban_sound_datasets_from_args():
    parser = argparse.ArgumentParser(description='Execute the strawman approach.')
    parser.add_argument('val_fold', metavar='N', type=int, help='the fold to use for validation')
    parser.add_argument('test_fold', metavar='N', type=int, help='the fold to use for testing')

    args = parser.parse_args()

    training, validation, testing = get_urban_sound_datasets_with_validation(
        validation_fold=args.val_fold, test_fold=args.test_fold
    )

    return training, validation, testing


def generate_data():
    t_loader, v_loader = get_urban_sound_datasets()

    def scenario_function(digit, last_digits, threshold, available_digits):
        if digit == last_digits[-1]:
            return digit, True

        return None, None

    gather_examples(
        t_loader, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'sounds_train_data.txt',
        'init_sound_train_data.txt', get_true_values=sound_true_values, network_clause='sound',
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        relevant_digits=1, scenario_function=scenario_function, threshold=None
    )
    gather_examples(
        v_loader, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'sounds_test_data.txt',
        'init_sound_test_data.txt', get_true_values=sound_true_values, network_clause='sound',
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        relevant_digits=1, scenario_function=scenario_function, threshold=None
    )


if __name__ == '__main__':
    generate_data()
