from examples.NIPS.MNIST.noisy_sequence_detection.scenarios100.scenario100_2.generate_data import generate_data
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values, \
    get_urban_sound_datasets_from_args


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit in last_digits:
        return digit, True

    return None, None


def make_audio_scenario100_for_window(window):
    training, validation, testing = get_urban_sound_datasets_from_args()

    generate_data(
        noises_function=lambda: [0.0],
        scenario_function=get_correct_digit_for_initiated_at,
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
    make_audio_scenario100_for_window(1)
