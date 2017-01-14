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
- jupyter
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
    export ICTDIR=`pwd`

    echo setting ICTDIR
    export ICDIR=$ICTDIR/invisible_cities/

    echo setting PYTHONPATH
    export PYTHONPATH=$ICTDIR:$PYTHONPATH
}

function download_test_db {
    echo Downloading database
    python $ICDIR/database/download.py
}

function compile_cython_components {
    python setup.py develop
}

function compile_and_test {
    compile_cython_components
    run_tests
}

function clean {
    echo "Cleaning IC generated files:"
    C_FILES=`find . -name '*.c'`
    SOFILES=`find . -name '*.so'`
    REMOVE="$C_FILES $SOFILES"
    if [ ! -z "${REMOVE// }" ]
    then
        for FILE in $REMOVE
        do
            COMMAND="rm $FILE"
            echo $COMMAND
            $COMMAND
        done
    else
        echo Nothing found to clean
    fi
}

## Main command dispatcher

THIS=manage.sh

case $COMMAND in
    install_and_check)      install_and_check ;;
    install)                install ;;
    work_in_python_version) work_in_python_version ;;
    make_environment)       make_environment ;;
    run_tests)              run_tests ;;
    compile_and_test)       compile_and_test ;;
    download_test_db)       download_test_db ;;
    clean)                  clean ;;
    
    *) echo Unrecognized command: ${COMMAND}
       echo
       echo Usage:
       echo
       echo "source $THIS install_and_check X.Y"
       echo "source $THIS install X.Y"
       echo "source $THIS work_in_python_version X.Y"
       echo "bash   $THIS make_environment X.Y"
       echo "bash   $THIS run_tests"
       echo "bash   $THIS compile_and_test"
       echo "bash   $THIS download_test_db"
       echo "bash   $THIS clean"
       ;;
esac
