import os

import numpy  as np
import tables as tb

from pytest   import       fixture
from pytest   import          mark

from . rwf_io import    rwf_writer
from . rwf_io import buffer_writer


@mark.parametrize("group_name", (None, 'RD', 'BLR'))
def test_rwf_writer(config_tmpdir, group_name):

    nevt       =       3
    nsensor    =       3
    nsample    =      10
    table_name = 'testwf'

    ofile      = os.path.join(config_tmpdir, 'testRWF.h5')

    test_data  = np.random.randint(10, size = (nevt, nsensor, nsample))

    with tb.open_file(ofile, "w") as h5out:
        rwf_writer_ = rwf_writer(h5out,
                                 group_name      = group_name,
                                 table_name      = table_name,
                                 n_sensors       =    nsensor,
                                 waveform_length =    nsample)

        for evt in test_data:
            rwf_writer_(evt)

    with tb.open_file(ofile) as h5test:
        if group_name is None:
            group = h5test.root
        else:
            group = getattr(h5test.root, group_name)


        assert table_name in group

        table = getattr(group, table_name)
        assert table.shape == (nevt, nsensor, nsample)
        assert np.all(test_data == table.read())


@mark.parametrize("event triggers".split(),
                  ((2, [10]), (3, [10, 1100]), (4, [20])))
def test_buffer_writer(config_tmpdir, event, triggers):

    run_number  = -6400
    n_pmt       =    12
    nsamp_pmt   =   100
    n_sipm      =  1792
    nsamp_sipm  =    10

    buffers = [(np.random.poisson(5, (n_pmt , nsamp_pmt )),
                np.random.poisson(5, (n_sipm, nsamp_sipm))) for _ in triggers]

    out_name = os.path.join(config_tmpdir, 'test_buffers.h5')
    with tb.open_file(out_name, 'w') as data_out:

        buffer_writer_ = buffer_writer(data_out,
                                       run_number = run_number,
                                       n_sens_eng =      n_pmt,
                                       n_sens_trk =     n_sipm,
                                       length_eng =  nsamp_pmt,
                                       length_trk = nsamp_sipm)

        buffer_writer_(event, triggers, buffers)

    pmt_wf  = np.array([b[0] for b in buffers])
    sipm_wf = np.array([b[1] for b in buffers])
    with tb.open_file(out_name) as h5saved:
        assert 'Run'      in h5saved.root
        assert 'pmtrd'    in h5saved.root
        assert 'sipmrd'   in h5saved.root
        assert 'events'   in h5saved.root.Run
        assert 'runInfo'  in h5saved.root.Run
        assert 'eventMap' in h5saved.root.Run

        nsaves = len(triggers)
        assert len(h5saved.root.Run.events  ) == nsaves
        assert len(h5saved.root.Run.eventMap) == nsaves
        assert len(h5saved.root.Run.runInfo ) == nsaves
        assert np.all([r[0] == run_number for r in h5saved.root.Run.runInfo])

        assert h5saved.root.pmtrd .shape == (nsaves, n_pmt ,  nsamp_pmt)
        assert np.all(h5saved.root.pmtrd [:] ==  pmt_wf)

        assert h5saved.root.sipmrd.shape == (nsaves, n_sipm, nsamp_sipm)
        assert np.all(h5saved.root.sipmrd[:] == sipm_wf)
