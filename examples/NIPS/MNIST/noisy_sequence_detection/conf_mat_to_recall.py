import click
import numpy as np


def calculate_recalls(matrices, c):
    recalls = []

    for matrix in matrices:
        recalls.append(matrix[c, c] / sum(matrix[c, :]))

    return recalls


@click.command()
@click.argument('conf_mat_path', required=True)
@click.argument('digit')
def conf_mat_to_recall(conf_mat_path, digit):
    digit = int(digit)

    in_digit = False

    with open(conf_mat_path, 'r') as f:
        initiatedAts = []
        digits = []

        current_matrix = []

        for l in f:
            if l.startswith('scenario'):
                pass
            elif l.startswith('digit'):
                if current_matrix:
                    initiatedAts.append(np.array(current_matrix))
                    current_matrix = []

                in_digit = True
            elif l.startswith('initiatedAt'):
                if current_matrix:
                    digits.append(np.array(current_matrix))
                    current_matrix = []

                in_digit = False
            else:
                line = map(int, l.split(','))
                current_matrix.append(list(line))

        if current_matrix:
            if in_digit:
                digits.append(np.array(current_matrix))
            else:
                initiatedAts.append(np.array(current_matrix))

    if digit % 2 == 0:
        print(','.join(map(str, calculate_recalls(initiatedAts, int(digit / 2 + 6)))))
    else:
        print(','.join(map(str, calculate_recalls(initiatedAts, int(digit / 2 + 1)))))
    print(','.join(map(str, calculate_recalls(digits, digit))))


if __name__ == '__main__':
    conf_mat_to_recall()

