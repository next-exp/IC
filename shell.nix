# To update `commit-id` go to https://status.nixos.org/, which lists the latest
# commit that passes all the tests for any release. Unless there is an
# overriding reason, pick the latest stable NixOS release, at the time of
# writing this is nixos-20.03.
let
  commit-id = "f1a79c86358c5464c64b4fad00fca07a10e62a74";
  nixpkgs-url = "https://github.com/nixos/nixpkgs/archive/${commit-id}.tar.gz";
  pkgs = import (builtins.fetchTarball { url = nixpkgs-url; }) {};
  python = pkgs.python37;
  pypkgs = python.pkgs;
  local = {
    # pytest-instafail was unavailable in nixpkgs at time of writing
    pytest-instafail = pypkgs.buildPythonPackage rec {
      pname = "pytest-instafail";
      version = "0.4.2";
      src = pypkgs.fetchPypi {
        inherit pname version;
        sha256 = "10lpr6mjcinabqynj6v85bvb1xmapnhqmg50nys1r6hg7zgky9qr";
      };
      buildInputs = [ pypkgs.pytest ];
    };
  };

  command = pkgs.writeShellScriptBin;

in

pkgs.mkShell {
  pname = "invisible-cities";
  buildInputs = [
    pkgs.git
    pypkgs.cython
    pypkgs.jupyter
    pypkgs.jupyterlab
    pypkgs.matplotlib
    pypkgs.networkx
    pypkgs.notebook
    pypkgs.numpy
    pypkgs.pandas
    pypkgs.seaborn
    pypkgs.pymysql
    pypkgs.tables
    pypkgs.scipy
    pypkgs.sphinx
    pypkgs.tornado
    pypkgs.pytest
    pypkgs.flaky
    pypkgs.hypothesis
    pypkgs.pytest_xdist
    local.pytest-instafail

    (command "ic-compile"  "python setup.py build_ext --inplace")
    (command "ic-test"     "pytest --instafail --no-success-flaky-report")
    (command "ic-test-par" ''
      N_PROC=$1
      STATUS=0
      pytest --instafail -n ''${N_PROC:-auto} -m "not serial" --no-success-flaky-report || STATUS=$?
      pytest --instafail                      -m      serial  --no-success-flaky-report || STATUS=$?
      [[ $STATUS = 0 ]]
    '')
    (command "ic-clean" ''
      echo "Cleaning IC generated files:"
      FILETYPES='*.c *.so *.pyc __pycache__'
      for TYPE in $FILETYPES
      do
          echo Cleaning $TYPE files
          REMOVE=`find . -name $TYPE`
          if [ ! -z ''${REMOVE// } ]
          then
              for FILE in $REMOVE
              do
                 rm -rf $FILE
              done
          else
              echo Nothing found to clean in $TYPE
          fi
      done
    '')
    (command "ic-admin-download-database" ''
      echo Downloading database
      python $ICDIR/database/download.py $1
    '')
  ];

  shellHook = ''
    export ICTDIR=`pwd`
    export ICDIR=$ICTDIR/invisible_cities/
    export PYTHONPATH=$ICTDIR:$PYTHONPATH
    export NBDIR=`pwd`
    export IC_NOTEBOOK_DIR=$NBDIR/invisible_cities/
    export PATH=$ICTDIR/bin:$PATH

    ic-compile
  '';

}