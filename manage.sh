#!/usrbin/env bash

COMMAND=$1
PYTHON_VERSION=$2

echo ${PYTHON_VERSION}

function install_and_check {
    install
    run_tests
}

function install {
    install_conda
    make_environment
    python_version_env
    compile_cython_components
}

function install_conda {
    # Identifies your operating system and installs the appropriate
    # version of conda. We only consider Linux and OS X at present.

    # Does nothing if conda is already found in your PATH.

    case "$(uname -s)" in

        Darwin)
            export CONDA_OS=MacOSX
            ;;

        Linux)
            export CONDA_OS=Linux
            ;;

        # CYGWIN*|MINGW32*|MSYS*)
        #   echo 'MS Windows'
        #   ;;

        *)
            echo "Installation only supported on Linux and OS X"
            exit 1
            ;;
    esac

    if which conda ; then
        echo Conda already installed. Skipping conda installation.
    else
        echo Installing conda for $CONDA_OS
        wget http://repo.continuum.io/miniconda/Miniconda-latest-${CONDA_OS}-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        export PATH="$HOME/miniconda/bin:$PATH"
        echo Prepended $HOME/miniconda/bin to your PATH. Please add it to your shell profile for future sessions.
        conda config --set always_yes yes --set changeps1 no
        conda update conda
        echo Prepended $HOME/miniconda/bin to your PATH. Please add it to your shell profile for future sessions.
    fi
}

function make_environment {
    YML_FILENAME=environment${PYTHON_VERSION}.yml

    echo creating ${YML_FILENAME}

    cat <<EOF > ${YML_FILENAME}
name: IC${PYTHON_VERSION}
dependencies:
- python=${PYTHON_VERSION}
- cython
- numpy
- pandas
- pymysql
- pytables
- pytest
- scipy
- matplotlib
- pip:
  - hypothesis-numpy
  - flaky
EOF

    conda env create -f ${YML_FILENAME}
}

function python_version_env {
    # Activate the relevant conda env
    source activate IC${PYTHON_VERSION}
    # Set IC environment variables and download database
    ic_env
}

function work_in_python_version {
    python_version_env
    compile_cython_components
    run_tests
}

function run_tests {
    # Ensure that the test database is present
    if [ ! -f invisible_cities/database/localdb.sqlite3 ]; then
        download_test_db
    fi

    # Run the test suite
    pytest
}

function ic_env {
    echo setting ICDIR
    export ICDIR=`pwd`

    echo setting ICTDIR
    export ICTDIR=$ICDIR/invisible_cities/

    echo setting PYTHONPATH
    export PYTHONPATH=$ICDIR:$PYTHONPATH
}

function download_test_db {
    echo Downloading database
    python $ICTDIR/database/download.py
}

function compile_cython_components {
    python setup.py develop
}

function compile_and_test {
    compile_cython_components
    run_tests
}

## Main command dispatcher

case $COMMAND in
    install_and_check)      install_and_check ;;
    install)                install ;;
    work_in_python_version) work_in_python_version ;;
    make_environment)       make_environment ;;
    run_tests)              run_tests ;;
    compile_and_test)       compile_and_test ;;
    download_test_db)       download_test_db ;;

    *) echo Unrecognized command: ${COMMAND}
       echo
       echo Usage:
       echo
       echo source $0 install_and_check X.Y
       echo soruce $0 install X.Y
       echo source $0 work_in_python_version X.Y
       echo bash $0 make_environment X.Y
       echo bash $0 run_tests
       echo bash $0 compile_and_test
       echo bash $0 download_test_db
       exit 1
       ;;
esac
