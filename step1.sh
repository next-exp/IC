#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: step1.sh git_repo_path"
        exit
fi

DSTFOLDER=$1

UPSTREAM="https://github.com/jmbenlloch/IC-1.git"
LFSOBJECTS="https://jobenllo.web.cern.ch/jobenllo/lfs_objects.tar.bz2"

git clone $UPSTREAM $DSTFOLDER
cd $DSTFOLDER/.git/lfs

if which wget; then
        wget $LFSOBJECTS
else
        curl $LFSOBJECTS -o `basename $LFSOBJECTS`
fi

rm -r objects
tar xvf lfs_objects.tar.bz2
rm lfs_objects.tar.bz2
