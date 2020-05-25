"""
-----------------------------------------------------------------------
                                Isidora
-----------------------------------------------------------------------

From ancient Greek, Ἰσίδωρος: gift of Isis.

This city removes the signal-derivative effect of the PMT waveforms.
"""
from functools import partial

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import sensor_data
from .  components import WfType
from .  components import wf_from_files
from .  components import deconv_pmt

import tables as tb

from .. dataflow          import dataflow as fl
from .. dataflow.dataflow import push
from .. dataflow.dataflow import pipe
from .. dataflow.dataflow import fork
from .. dataflow.dataflow import sink

from .. reco                import tbl_functions as tbl
from .. io.          rwf_io import           rwf_writer
from .. io.run_and_event_io import run_and_event_writer




@city
def isidora(files_in, file_out, compression, event_range, print_mod,
            detector_db, run_number, n_baseline):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.

    """
    sd = sensor_data(files_in[0], WfType.rwf)

    rwf_to_cwf = fl.map(deconv_pmt(detector_db, run_number, n_baseline), args="pmt", out="cwf")

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        RWF        = partial(rwf_writer, h5out, group_name='BLR')
        write_pmt  = sink(RWF(table_name='pmtcwf' , n_sensors=sd.NPMT , waveform_length=sd.PMTWL ), args="cwf" )
        write_sipm = sink(RWF(table_name='sipmrwf', n_sensors=sd.NSIPM, waveform_length=sd.SIPMWL), args="sipm")

        write_event_info_ = run_and_event_writer(h5out)

        write_event_info = sink(write_event_info_, args=("run_number", "event_number", "timestamp"))

        event_count = fl.spy_count()

        evtnum_collect = collect()

        result = push(source = wf_from_files(files_in, WfType.rwf),
                      pipe   = pipe(fl.slice(*event_range, close_all=True),
                                    event_count.spy,
                                    print_every(print_mod),
                                    fl.branch("event_number", evtnum_collect.sink),
                                    fork((rwf_to_cwf, write_pmt       ),
                                        (             write_sipm      ),
                                        (             write_event_info))),
                      result = dict(events_in   = event_count   .future,
                                    evtnum_list = evtnum_collect.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
