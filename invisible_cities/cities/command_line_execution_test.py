from os import getenv
from os.path import join
from subprocess import run
from pytest import mark

@mark.slow
@mark.parametrize('city',
                  'diomira isidora irene dorothea zaira'.split())
def test_command_line_run(city, tmpdir_factory):
    ICTDIR = getenv('ICTDIR')
    # Use the example config file included in the repository
    config_file_name = join(ICTDIR, 'invisible_cities/config', city+'.conf')
    # Ensure that output does not pollute: send it to a temporary dir
    temp_dir = tmpdir_factory.mktemp('output_files')
    out_file_name = join(temp_dir, city+'.out')
    # The actual command that we want to test
    command = ('{city} -c {config_file_name} -o {out_file_name} -n 1'
               .format(**locals()))
    r = run(command, shell = True)
    assert r.returncode == 0

