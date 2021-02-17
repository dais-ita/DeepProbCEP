import random

import click

from examples.NIPS.MNIST.noisy_sequence_detection.balanced_reduce import read_init_file, write_init_file


def get_balanced_oversampling(classes, total, options):
    chosen = []
    chosen_by_class = dict.fromkeys(classes, 0)

    i = 0
    while i < total:
        # chosen_class = random.choice(classes)  # Choosing one of the classes at random for each iteration. Use this
        # for a random distribution between the classes (will tend to be roughly balanced as we are equally likely to
        # choose any class at every iteration, but classes are unlikely to have *exactly* the same number of instances)

        chosen_class = classes[(i + 1) % len(classes)]  # Iterating over each of the classes for each iteration. Use
        # this to get a perfectly balanced distribution between the classes (the biggest possible difference between
        # class instances will be 1, which will happen if total / len(classes) is not an integer)

        index = chosen_by_class[chosen_class] % len(options[chosen_class])

        chosen.append(options[chosen_class][index])

        chosen_by_class[chosen_class] += 1

        i += 1

    return chosen, chosen_by_class


@click.command()
@click.argument('filename', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt')
@click.argument('total', default=5000)
@click.argument('output', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt_')
@click.option('--skip', default=5)
def oversample(filename, total, output, skip):
    options = read_init_file(filename, skip)

    classes = list(options.keys())

    chosen, chosen_by_class = get_balanced_oversampling(classes, total, options)

    write_init_file(output, chosen)

    print(chosen_by_class)


if __name__ == '__main__':
    oversample()
