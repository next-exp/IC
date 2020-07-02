IC: Invisible Cities
==============================================

IC stands for Invisible Cities and also for Italo Calvino, the author of the master piece.

Quickstart guide
----------------

+ Make sure that `Nix is installed <doc/nix/install-nix/README.md>`_ on your machine

+ Make sure that `home-manager is installed <doc/nix/home-manager/README.md>`_ for your user

+ Clone the repository

+ :code:`cd` into the `IC` directory you have just cloned

+ :code:`direnv allow`

+ :code:`ic-test-par`

The last step launches an exhaustive set of IC tests. If these pass, you can be
confident that everything has been configured as it should.

The :code:`cd /path/to/IC` step is the only one you will have to repeat, on a
machine on which these steps have already been carried out, in order to resume
work on IC.

To check your progress when you are developing, you will want to compile any
Cython components that have changed and run the test suite. This can be done
with

.. code-block::

   ic-compile && ic-test-par

If you check out a commit which requires an older set of dependencies, Nix,
`home-manager` and `direnv` will together ensure that you automatically switch
to an appropriate environment, creating it on the fly if necessary, without you
having do do anything at all, other than maybe waiting for downloads of older
packages to complete.

:Travis CI: |travis|

.. |travis| image:: https://img.shields.io/travis/nextic/IC.png
        :target: https://travis-ci.org/nextic/IC
