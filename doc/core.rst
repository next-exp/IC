Core modules
============

A major revision of the core modules has been conducted. This includes
cleaning up, eliminating many obsolete functions, as well as
re-organizing functions, across modules.

- **core_functions.py**: general purpose functions. It has been
  cleaned up, and a simple test suite has been added. There is also a
  doc notebook (which is not included in this commit, waiting for
  policy on notebooks and docs to be fully agreed).

- **core_functions_performance.py**: a module with a simple example
  case of performance of python vs Cython for a particular example.

- **mpl_functions.py**: general purpose plotting functions. No tests
  are relevant here, but the plotting functions are exercised in the
  documentation notebooks (diomira, irene) which are ready to be
  committed.

- **system_of_units.py**: definition of system of units. Extensively
  tried and tested, stable.

- **random_sampling.py**: sampler of SiPM pdf
  functions. Diomira-notebook shows that it behaves as expected but a
  test would be nice (Gonzalo Martinez, pending).

- **log_config.py**: configures the logger. Not used currently,
  possible candidate for deletion.

- **configure.py**: utility used for configuration, works as expected,
  stable.

- **sensor_functions.py**: collects functions that manipulate
  sensors. Most of them exercised in notebooks. A test could be added.

- **tbl_functions.py**: utility for pYtables manipulation.

- **wfm_functions.py**: waveform manipulation. There is a simple test
  suite (can be enlarged).

- **wfm_to_df_functions.py**: collection of functions where wmf are
  transformed to data frames for manipulation. Not used at the
  moment. Possible candidate for deletion.

- **mctrk_functions.py**: for manipulation of MC "true" tracks. Not
  being used at the moment (reconstruction does not use "true"
  variables) but eventually useful.

- **peak_functions.py**: to be revised in next cycle, associated to
  Irene.

- **pmaps_functions.py**: to be revised in next cycle, associated to
  Irene.
