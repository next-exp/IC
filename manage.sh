#!/usrbin/env bash

COMMAND=$1
ARGUMENT=$2


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
        if which wget; then
            wget https://repo.continuum.io/miniconda/Miniconda3-4.3.21-${CONDA_OS}-x86_64.sh -O miniconda.sh
        else
            curl https://repo.continuum.io/miniconda/Miniconda3-4.3.21-${CONDA_OS}-x86_64.sh -o miniconda.sh
        fi
        bash miniconda.sh -b -p $HOME/miniconda
        export PATH="$HOME/miniconda/bin:$PATH"
        echo Prepended $HOME/miniconda/bin to your PATH. Please add it to your shell profile for future sessions.
        conda config --set always_yes yes --set changeps1 no
        echo Prepended $HOME/miniconda/bin to your PATH. Please add it to your shell profile for future sessions.
    fi
}

function make_environment {
    YML_FILENAME=environment${PYTHON_VERSION}.yml

    echo creating ${YML_FILENAME}

    cat <<EOF > ${YML_FILENAME}
name: IC3.6
dependencies:
- cython=0.26=py36_0
- jupyter=1.0.0=py36_3
- matplotlib=2.0.2=np113py36_0
- networkx=1.11=py36_0
- notebook=5.0.0=py36_0
- numpy=1.13.1=py36_0
- pandas=0.20.3=py36_0
- pymysql=0.7.9=py36_0
- pytables=3.4.2=np113py36_0
- pytest=3.2.1=py36_0
- python=3.6.2=0
- scipy=0.19.1=np113py36_0
- sphinx=1.6.3=py36_0
- pip:
  - flaky==3.4.0
  - hypothesis==3.32.0
  - pytest-xdist==1.20.0
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
    pytest -v --no-success-flaky-report
}

function run_tests_par {
    # Ensure that the test database is present
    if [ ! -f invisible_cities/database/localdb.sqlite3 ]; then
        download_test_db
    fi

    # Run the test suite
    EXIT=0
    pytest -v -n ${N_PROC} -m "not serial" --no-success-flaky-report || EXIT=$?
    pytest -v              -m      serial  --no-success-flaky-report || EXIT=$?
    exit $EXIT
}

function ic_env {
    export ICTDIR=`pwd`
    echo ICDIR set to $ICTDIR

    export ICDIR=$ICTDIR/invisible_cities/
    echo ICDIR set to $ ICDIR

    export PYTHONPATH=$ICTDIR
    echo PYTHONPATH set to $PYTHONPATH

    export NBDIR=`pwd`
    echo NBDIR set to $NBDIR

    export IC_NOTEBOOK_DIR=$NBDIR/invisible_cities/
    echo IC_NOTEBOOK_DIR set to $ IC_NOTEBOOK_DIR

    export PATH=$ICTDIR/bin:$PATH
    echo $ICTDIR/bin added to path
}

function show_ic_env {
    conda-env list

    echo "ICTDIR set to $ICTDIR"
    echo "ICDIR  set to $ICDIR"
    echo PYTHONPATH set to $PYTHONPATH
}

function download_test_db {
    echo Downloading database
    python $ICDIR/database/download.py
}

function download_test_db_dev {
    echo Downloading dev database
    python $ICDIR/database/download.py NEWDB_dev
}

function compile_cython_components {
    python setup.py develop
}

function compile_and_test {
    compile_cython_components
    run_tests
}

function compile_and_test_par {
    compile_cython_components
    run_tests_par
}

function clean {
    echo "Cleaning IC generated files:"
    FILETYPES='*.c *.so *.pyc __pycache__'
    for TYPE in $FILETYPES
    do
		echo Cleaning $TYPE files
        REMOVE=`find . -name $TYPE`
        if [ ! -z "${REMOVE// }" ]
        then
            for FILE in $REMOVE
            do
               rm -rf $FILE
            done
        else
            echo Nothing found to clean in $TYPE
        fi
    done
}

THIS=manage.sh

## Interpret meaning of command line argument depending on which
## function will receive it.

case $COMMAND in
    run_tests_par | compile_and_test_par)     N_PROC=${ARGUMENT:-auto} ;;
    *)                                PYTHON_VERSION=${ARGUMENT}       ;;
esac

## Main command dispatcher

case $COMMAND in
    install_and_check)      install_and_check ;;
    install)                install ;;
    work_in_python_version) work_in_python_version ;;
    make_environment)       make_environment ;;
    run_tests)              run_tests ;;
    run_tests_par)          run_tests_par ;;
    compile_and_test)       compile_and_test ;;
    compile_and_test_par)   compile_and_test_par ;;
    download_test_db)       download_test_db ;;
    download_test_db_dev)   download_test_db_dev ;;
    clean)                  clean ;;
    show_ic_env)            show_ic_env ;;

    *) echo Unrecognized command: ${COMMAND}
       echo
       echo Usage:
       echo
       echo "source $THIS install_and_check X.Y"
       echo "source $THIS install X.Y"
       echo "source $THIS work_in_python_version X.Y"
       echo "bash   $THIS make_environment X.Y"
       echo "bash   $THIS run_tests"
       echo "bash   $THIS run_tests_par"
       echo "bash   $THIS compile_and_test"
       echo "bash   $THIS compile_and_test_par"
       echo "bash   $THIS download_test_db"
       echo "bash   $THIS download_test_db_dev"
       echo "bash   $THIS clean"
       echo "bash   $THIS show_ic_env"
       ;;
esac
