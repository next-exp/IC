IC: Invisible Cities
==============================================

IC stands for Invisible Cities and also for Italo Calvino, the author of the master piece. 

Quickstart guide
----------------

If you have just cloned the repository for the first time, then issue
the command

.. code-block::

  source manage.sh install_and_check 3.5

Where the 3.5 can be replaced with any sensible Python version you
like. (On 2017-02-24 we actively support Python 3.6, 3.5 and 2.7.)
   
If you have already done the above procedure once, then you should
already have an `IC3.5` conda environment available, as long as
${HOME}/miniconda3/bin (or an earlier conda installation) is in your
`PATH`. (You may like to add the location of your conda installation
to your shell startup file.) To start working in an IC environment you
set up earlier issue the command

.. code-block::

  source manage.sh work_in_python_version 3.5

(replacing 3.5 with whatever python version is relevant for your
case.)

If you wish to develop and test in a python version in which you have
not worked on IC before, you will need to create the corresponding
conda environment:

.. code-block::

  source manage.sh make_environment 2.7

(replacing 2.7 with whatever python version is relevant for your
case.) After this you will be able to work in that environment by
selecting it as before

.. code-block::

  source manage.sh work_in_python_version 2.7

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
:Coveralls: |coveralls|

.. |travis| image:: https://img.shields.io/travis/nextic/IC.png
        :target: https://travis-ci.org/nextic/IC

.. |coveralls| image:: https://coveralls.io/repos/nextic/IC/badge.png
        :target: https://coveralls.io/r/nextic/IC
