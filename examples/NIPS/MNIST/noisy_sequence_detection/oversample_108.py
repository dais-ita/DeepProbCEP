import random

import click

from examples.NIPS.MNIST.noisy_sequence_detection.balanced_reduce import write_init_file
from examples.NIPS.MNIST.noisy_sequence_detection.oversample import get_balanced_oversampling


def read_init_file_108(filename, skip, swap_from_class):
    options = {}

    skipped = 0
    with open(filename, 'r') as f:
        for line in f:
            if skipped < skip:
                skipped += 1
                continue

            if line.startswith('initiatedAtNoise('):
                line_class = swap_from_class
            else:
                line_class = line.split('(', 1)[1].split(',')[0]

            options.setdefault(line_class, [])
            options[line_class].append(line)

    return options


def oversample_with_swap_class(filename, total, output, skip, digit_classes, swap_from_class):
    options = read_init_file_108(filename, skip, swap_from_class)

    classes = list(set(list(options.keys()) + digit_classes))

    chosen, chosen_by_class = get_balanced_oversampling(classes, total, options)

    write_init_file(output, chosen)

    print(chosen_by_class)

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

    swap_file = filename[:-30] + 'swap.txt'
    with open(swap_file, 'r') as f:
        swap_from = int(f.readline().strip().split(',')[0])

    swap_from_class = digit_classes[swap_from]

    oversample_with_swap_class(filename, total, output, skip, digit_classes, swap_from_class)


if __name__ == '__main__':
    oversample()
