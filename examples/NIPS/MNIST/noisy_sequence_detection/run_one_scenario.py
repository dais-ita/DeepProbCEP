import sys

from examples.NIPS.MNIST.mnist import neural_predicate
from examples.NIPS.MNIST.noisy_sequence_detection.cep_deep_pre_comp import CEPDeepPreComp
from examples.NIPS.MNIST.noisy_sequence_detection.run import add_files_to, make_train_val_test_models, \
    get_problog_file_for, my_validation, make_problog_strings

sys.path.append('../../../')
from train import epoch_train_model
from data_loader import load
import torch

torch.manual_seed(42)


def run_one_scenario(training_data, validation_data, test_data, problog_files, problog_train_files, problog_val_files,
                     problog_test_files, window, use_validation=False):
    queries = load(training_data)
    validation_queries = load(validation_data)
    test_queries = load(test_data)

    problog_train_string, problog_val_string, problog_test_string = make_problog_strings(
        problog_files, problog_test_files, problog_train_files, problog_val_files
    )

    # precomp = CEPDeepPreComp(training_data, problog_train_files[0], window=window)
    precomp = None
    # val_precomp = CEPDeepPreComp(validation_data, problog_val_files[0], window=window)
    val_precomp = None

    model_to_train, model_to_val, model_to_test, optimizer = make_train_val_test_models(
        neural_predicate, problog_train_string, problog_val_string, problog_test_string, training_caching=False,
        precomp=precomp, val_precomp=val_precomp
    )

    model_to_train.load_state('testing_snapshots/initial_state.mdl')

    _, best_epoch, best_weights_fname = epoch_train_model(
        model_to_train,
        queries,
        5,
        optimizer,
        # validation=lambda _: my_validation(
        #     model_to_val,
        #     validation_queries
        # ),
        validation=None,
        patience=2,
        log_epoch=1,
        snapshot_name='testing_snapshots/model_',
        shuffle=False
    )

    # my_test(
    #     model_to_test, test_queries, test_functions={
    #         'mnist_net': lambda *args, **kwargs: neural_predicate(
    #             *args, **kwargs, dataset='test'
    #         )
    #     }
    # )


if __name__ == '__main__':
    directory = 'testing'
    folder = 'scenario100_15_100'
    subfolder = 'noise_0_00'
    window_size = 15

    prob_ec_cached = get_problog_file_for(directory, folder, 'prob_ec_cached.pl')
    event_defs = get_problog_file_for(directory, folder, 'event_defs.pl')

    run_one_scenario(
        training_data='{}/{}/{}/init_train_data_clean.txt'.format(directory, folder, subfolder),
        validation_data='{}/{}/{}/init_val_data.txt'.format(directory, folder, subfolder),
        test_data='{}/{}/{}/init_digit_test_data.txt'.format(directory, folder, subfolder),
        problog_files=[
            prob_ec_cached,
            event_defs
        ],
        problog_train_files=[
            '{}/{}/{}/in_train_data.txt'.format(directory, folder, subfolder)
        ],
        problog_val_files=[
            '{}/{}/{}/in_val_data.txt'.format(directory, folder, subfolder)
        ],
        problog_test_files=[
            '{}/{}/{}/in_test_data.txt'.format(directory, folder, subfolder)
        ],
        window=window_size
    )
