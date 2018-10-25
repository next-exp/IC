from os      import getenv
from os.path import join

from subprocess import check_output
from subprocess import CalledProcessError
from subprocess import STDOUT

from pytest import mark

@mark.slow
# @mark.parametrize('city',
#                   'diomira isidora irene dorothea zaira penthesilea'.split())
# TODO understand what's wrong with isidora (in Travis)
@mark.parametrize('city',
                  'diomira isidora irene dorothea penthesilea zaira zemrude phyllis moriana'.split())

def test_command_line_run(city, tmpdir_factory):
    ICTDIR = getenv('ICTDIR')
    # Use the example config file included in the repository
    config_file_name = join(ICTDIR, 'invisible_cities/config/', f'{city}.conf')
    # Ensure that output does not pollute: send it to a temporary dir
    temp_dir = tmpdir_factory.mktemp('output_files')
    out_file_name = join(temp_dir, f'{city}.out')
    # The actual command that we want to test
    command = ('city {city} {config_file_name} -o {out_file_name}'
               .format(**locals()))
    try:
        check_output(command, shell = True, stderr=STDOUT)
    except CalledProcessError as e:
        # Ensure that stdout and stderr are visible when test fails
        print(e.stdout.decode())
        raise
