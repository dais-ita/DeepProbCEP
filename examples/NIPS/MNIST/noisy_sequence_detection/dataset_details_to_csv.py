import csv

import click
import os


def get_class(l):
    l = l.replace(' ', '')

    return l.split('(')[1].split(',')[0]


def get_details_for(f):
    file_classes = set()

    file_distribution = {}

    for l in f:
        line_class = get_class(l)

        file_classes.add(line_class)

        file_distribution.setdefault(line_class, 0)
        file_distribution[line_class] += 1

    return file_classes, file_distribution


@click.command()
@click.argument('directory', required=True, type=click.Path(exists=True))
@click.argument('output_file', required=True, type=click.Path())
def dataset_details_to_csv(directory, output_file):
    data_by_folder = {}

    classes = set()

    for folder in sorted(os.listdir(directory)):
        if folder.startswith('scenario'):
            data_by_folder[folder] = {}
            for subfolder in sorted(os.listdir('{}/{}'.format(directory, folder))):
                if subfolder.startswith('noise'):
                    with open('{}/{}/{}/init_train_data.txt'.format(directory, folder, subfolder), 'r') as f:
                        file_classes, file_distribution = get_details_for(f)

                        classes = classes.union(file_classes)

                        data_by_folder[folder][subfolder] = file_distribution

    with open(output_file, 'w') as f:
        csv_writer = csv.writer(f)

        classes_list = sorted(list(classes))

        csv_writer.writerow(['Directory', 'Folder', 'Subfolder', 'Total'] + classes_list)

        for folder in data_by_folder:
            for subfolder in data_by_folder[folder]:
                line = [directory, folder, subfolder, sum(data_by_folder[folder][subfolder].values())]
                line += list(
                    map(
                        lambda k: data_by_folder[folder][subfolder].get(k, 0),
                        classes_list
                    )
                )

                csv_writer.writerow(line)


if __name__ == '__main__':
    dataset_details_to_csv()
