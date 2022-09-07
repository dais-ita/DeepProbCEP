import random

import click

from examples.NIPS.MNIST.noisy_sequence_detection.balanced_reduce import write_init_file
from examples.NIPS.MNIST.noisy_sequence_detection.oversample import get_balanced_oversampling
from examples.NIPS.MNIST.noisy_sequence_detection.oversample_108 import read_init_file_108, oversample_with_swap_class


@click.command()
@click.argument('filename', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt')
@click.argument('total', default=5000)
@click.argument('output', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt_')
@click.option('--skip', default=5)
def oversample(filename, total, output, skip):
    digit_classes = [
        'sequence0 = true',
        'sequence0 = false',
        'sequence1 = true',
        'sequence1 = false',
        'sequence2 = true',
        'sequence2 = false',
        'sequence3 = true',
        'sequence3 = false',
        'sequence4 = true',
        'sequence4 = false',
    ]

    swap_from = 0

    swap_from_class = digit_classes[swap_from]

    oversample_with_swap_class(filename, total, output, skip, digit_classes, swap_from_class)


if __name__ == '__main__':
    oversample()
