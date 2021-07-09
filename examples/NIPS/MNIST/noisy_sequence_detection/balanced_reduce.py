from copy import deepcopy

import click
import random


def read_init_file(filename, skip):
    options = {}

    skipped = 0
    with open(filename, 'r') as f:
        for line in f:
            if skipped < skip:
                skipped += 1
                continue

            line_class = line.split('(', 1)[1].split(',')[0]

            options.setdefault(line_class, [])
            options[line_class].append(line)

    return options


def write_init_file(output, lines):
    with open(output, 'w') as f:
        for l in lines:
            f.write(l)


def get_balanced_reduction(classes, total, options):
    chosen = []
    chosen_by_class = dict.fromkeys(classes, 0)

    classes = deepcopy(classes)
    options = deepcopy(options)

    i = 0
    while i < total:
        chosen_class = random.choice(classes)

        if options[chosen_class]:
            chosen.append(options[chosen_class][0])
            options[chosen_class] = options[chosen_class][1:]

            chosen_by_class[chosen_class] += 1

            i += 1
        else:
            classes.remove(chosen_class)

    return chosen, chosen_by_class


@click.command()
@click.argument('filename', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt')
@click.argument('total', default=5000)
@click.argument('output', default='scenarios100/scenario100_2/noise_0_00/init_train_data.txt_')
@click.option('--skip', default=5)
def balanced_reduce(filename, total, output, skip):
    options = read_init_file(filename, skip)

    classes = list(options.keys())

    chosen, chosen_by_class = get_balanced_reduction(classes, total, options)

    write_init_file(output, chosen)

    print(chosen_by_class)


if __name__ == '__main__':
    balanced_reduce()
