IC: Invisible Cities
==============================================

IC stands for Invisible Cities and also for Italo Calvino, the author of the master piece. 

Quickstart guide
----------------

If you have just cloned the repository for the first time, then issue
the command

.. code-block::

  source manage.sh install_and_check 3.6

Where the 3.6 can be replaced with any sensible Python version you
like. (On 2017-06-27 we dropped support for Python 3.5, so Python 3.6
will be the only supported version until 3.7 is released and the third
party modules we use are uploaded to conda and pip.)
   
If you have already done the above procedure once, then you should
already have an `IC3.6` conda environment available, as long as
${HOME}/miniconda3/bin (or an earlier conda installation) is in your
`PATH`. (You may like to add the location of your conda installation
to your shell startup file.) To start working in an IC environment you
set up earlier issue the command

.. code-block::

  source manage.sh work_in_python_version 3.6

(replacing 3.6 with whatever python version is relevant for your
case.)

If you wish to develop and test in a python version in which you have
not worked on IC before, you will need to create the corresponding
conda environment:

.. code-block::

  source manage.sh make_environment 3.5

(replacing 3.5 with whatever python version is relevant for your
case.) After this you will be able to work in that environment by
selecting it as before

.. code-block::

  source manage.sh work_in_python_version 3.5

To check your progress when you are developing you will want to
compile Cython components and run the test suite. This can be done
with

.. code-block::

   bash manage.sh compile_and_test

If the test database changes, you will need to download the most
recent version:

.. code-block::

   bash manage.sh download_test_db
   

:Travis CI: |travis|

.. |travis| image:: https://img.shields.io/travis/nextic/IC.png
        :target: https://travis-ci.org/nextic/IC

