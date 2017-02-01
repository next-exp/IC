import pytest

@pytest.fixture(scope='session')
def irene_diomira_chain_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('irene_diomira_tests')

@pytest.fixture(scope='session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')
