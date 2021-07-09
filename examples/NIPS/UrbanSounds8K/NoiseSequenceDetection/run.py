import os
import re
import sys
from data_loader import load
import click

from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.UrbanSounds8K import neural_predicate_vggish
from train import epoch_train_model
from examples.NIPS.MNIST.noisy_sequence_detection.run import make_problog_strings
from examples.NIPS.UrbanSounds8K.SequenceDetection.run import run_linear, my_test
from examples.NIPS.UrbanSounds8K.sounds_utils import *
from model import Model
from optimizer import Optimizer
from network import Network

sys.path.append('../../../')


def audio_validation(model_to_validate, validation_queries):
    res = test(
        model_to_validate, validation_queries
    )

    return res[1][1]


def run_audio_with_validation(training_data, val_data, test_data, problog_files, problog_train_files=(),
                              problog_val_files=(), problog_test_files=(), neural_predicate=neural_predicate_vggish,
                              nn_model=SoundVGGish):
    scenario = training_data.split('/')[1]

    queries = load(training_data)
    val_queries = load(val_data)
    test_queries = load(test_data)

    problog_train_string, problog_val_string, problog_test_string = make_problog_strings(
        problog_files, problog_test_files, problog_train_files, problog_val_files
    )

    network = nn_model()
    net = Network(network, 'sound_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model_to_train = Model(problog_train_string, [net], caching=False)
    optimizer = Optimizer(model_to_train, 2)

    model_to_val = Model(problog_val_string, [net], caching=False)
    model_to_test = Model(problog_test_string, [net], caching=False)

    _, best_epoch, best_weights_fname = epoch_train_model(
        model_to_train,
        queries,
        100,
        # 1,
        optimizer,
        validation=lambda _: audio_validation(
            model_to_val,
            val_queries
        ),
        # validation=None,
        log_epoch=1,
        snapshot_name='snapshots/model_{}'.format(scenario),
        patience=10
    )

    model_to_test.load_state(best_weights_fname)

    my_test(
        model_to_test, test_queries
    )


@click.command()
@click.argument('start_path', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('--scenario', default='')
@click.option('--noise', default='')
def execute_scenarios(start_path, scenario, noise):
    for folder in sorted(os.listdir(start_path)):
        if folder.startswith('scenario') and re.search(scenario, folder):
            print("#######################################################################################")
            print(folder)

            prob_ec_cached = '{}/{}/prob_ec_cached.pl'.format(start_path, folder)
            if not os.path.isfile(prob_ec_cached):
                prob_ec_cached = 'ProbLogFiles/prob_ec_cached.pl'.format(start_path)

            event_defs = '{}/{}/event_defs.pl'.format(start_path, folder)
            if not os.path.isfile(event_defs):
                event_defs = 'ProbLogFiles/event_defs.pl'.format(start_path)

            for subfolder in sorted(os.listdir(start_path + folder)):
                # if subfolder != 'noise_1_00':
                #     continue

                if re.search(noise, subfolder) and os.path.isdir('{}/{}/{}'.format(start_path, folder, subfolder)):
                    print('===================================================================================')
                    print(subfolder)

                    run_audio_with_validation(
                        training_data='{}/{}/{}/init_train_data_clean.txt'.format(start_path, folder, subfolder),
                        val_data='{}/{}/{}/init_val_data.txt'.format(start_path, folder, subfolder),
                        test_data='{}/{}/{}/init_sound_test_data.txt'.format(start_path, folder, subfolder),
                        problog_files=[
                            prob_ec_cached,
                            event_defs
                        ],
                        problog_train_files=[
                            '{}/{}/{}/in_train_data.txt'.format(start_path, folder, subfolder)
                        ],
                        problog_val_files=[
                            '{}/{}/{}/in_val_data.txt'.format(start_path, folder, subfolder)
                        ],
                        problog_test_files=[
                            '{}/{}/{}/in_test_data.txt'.format(start_path, folder, subfolder)
                        ],
                        neural_predicate=neural_predicate_vggish,
                        nn_model=SoundVGGish
                    )


if __name__ == '__main__':
    execute_scenarios()
