import click

@click.command()
@click.argument('filename', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('out_filename', required=True, type=click.Path(writable=True))
@click.argument('scenario', default='')
def get_matrices(filename, out_filename, scenario):
    in_correct_scenario = False

    with open(filename, 'r') as f:
        with open(out_filename, 'w') as o:
            for l in f:
                if l.startswith('scenario{}'.format(scenario)):
                    in_correct_scenario = True
                    o.write(l)
                elif l.startswith('################'):
                    in_correct_scenario = False
                elif l.startswith('noise') and in_correct_scenario:
                    o.write(l)
                elif (l.startswith('digit') or l.startswith('sound') or l.startswith('initiatedAt')) and in_correct_scenario:
                    o.write(l)
                elif (l.startswith('[[') or l.startswith(' [')) and in_correct_scenario:
                    o.write(','.join(l.replace('[', '').replace(']', '').split()) + '\n')


if __name__ == '__main__':
    get_matrices()

