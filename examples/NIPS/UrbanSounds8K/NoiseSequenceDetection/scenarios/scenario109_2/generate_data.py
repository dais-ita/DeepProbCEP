from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.noisy_scenarios.scenario109_2.generate_data import \
    get_initiated_at_scenario109
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values, \
    get_urban_sound_datasets_from_args


def make_audio_scenario109_for_window(window):
    training, validation, testing = get_urban_sound_datasets_from_args()

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

    target = all_events[0]

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario109(
            *args, **kwargs, target=target
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=window,
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


if __name__ == '__main__':
    make_audio_scenario109_for_window(1)
