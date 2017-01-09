#!/bin/bash
#echo -n "Enter 0 to use an existing copy of DB or 1 to download a new copy --> "
#echo "your choice is $download"
export ICDIR=`pwd`
export ICTDIR=$ICDIR/invisible_cities/
export PYTHONPATH=$ICDIR:$PYTHONPATH
python $ICDIR/invisible_cities/database/download.py
