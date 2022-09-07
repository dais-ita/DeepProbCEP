import sys
from data_loader import load
import click

from examples.NIPS.prob_ec_testing import test
from train import epoch_train_model
from examples.NIPS.MNIST.noisy_sequence_detection.run import make_problog_strings, run_directory
from examples.NIPS.UrbanSounds8K.SequenceDetection.run import my_test
from examples.NIPS.UrbanSounds8K.sounds_utils import *
from model import Model
from optimizer import Optimizer
from network import Network

sys.path.append('../../../')


def audio_validation(model_to_validate, validation_queries, confusion_index=None):
    res = test(
        model_to_validate, validation_queries, confusion_index=confusion_index
    )

    print(res)

    return res[1][1]


def run_audio_with_validation(training_data, val_data, test_data, problog_files, problog_train_files=(),
                              problog_val_files=(), problog_test_files=(), neural_predicate=neural_predicate_vggish,
                              nn_model=SoundVGGish):
    raise Exception("Deprecated")
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
@click.option('--scenario', default='')
@click.option('--noise', default='')
@click.option('--directory', default='./scenarios100')
@click.option('--max_epochs', default=100)
@click.option('--audio_classes', default=10)
@click.option('--load_weights', default=None)
@click.option('--freeze_layers/--unfrozen_layers', default=False)
def execute_scenarios(scenario, noise, directory, max_epochs, audio_classes, load_weights, freeze_layers):
    if freeze_layers:
        nn_model = SoundVGGishFrozenLayers
    else:
        nn_model = SoundVGGish

    run_directory(
        directory=directory,
        scenario=scenario,
        noise=noise,
        load_weights=load_weights,
        nn_model=lambda: nn_model(n_classes=audio_classes),
        nn_name='sound_net',
        max_epochs=max_epochs,
        validation_function=audio_validation,
        test_function=my_test,
        neural_predicate=neural_predicate_vggish,
        class_type='sound',
    )


if __name__ == '__main__':
    execute_scenarios()
