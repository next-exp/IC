from functools   import wraps
from collections import Sequence
from argparse    import Namespace
from glob        import glob
from os.path     import expandvars
from itertools   import count
from enum        import Enum

import tables as tb
import numpy  as np

from .. dataflow               import dataflow      as fl
from .. evm .ic_containers     import SensorData
from .. database               import load_db
from .. sierpe                 import blr


def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        del conf.print_config_only
        del conf.hide_config
        del conf.no_overrides
        del conf.no_files
        del conf.full_files

        # TODO: we have decided to remove verbosity.
        # Needs to be removed form config parser
        del conf.verbosity

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        conf.files_in  = sorted(glob(expandvars(conf.files_in)))
        conf.file_out  =             expandvars(conf.file_out)

        conf.event_range  = _event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        return city_function(**vars(conf))
    return proxy


def _event_range(conf):
    if not hasattr(conf, 'event_range'):        return None, 1
    er = conf.event_range
    if not isinstance(er, Sequence): er = (er,)
    if len(er) == 1:                            return None, er[0]
    if len(er) == 2:                            return tuple(er)
    if len(er) == 0:
        raise ValueError('event_range needs at least one value')
    if len(er) >  2:
        raise ValueError('event_range accepts at most 2 values but was given {}'
                        .format(er))


def print_every(N):
    counter = count()
    return fl.branch(fl.map  (lambda _: next(counter), args="event_number", out="index"),
                     fl.slice(None, None, N),
                     fl.sink (lambda data: print(f"events processed: {data['index']}, event number: {data['event_number']}")))


def print_every_alternative_implementation(N):
    @fl.coroutine
    def print_every_loop(target):
        with fl.closing(target):
            for i in count():
                data = yield
                if not i % N:
                    print(f"events processed: {i}, event number: {data['event_number']}")
                target.send(data)
    return print_every_loop


# TODO: consider caching database
def deconv_pmt(run_number, n_baseline):
    DataPMT    = load_db.DataPMT(run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        return blr.deconv_pmt(RWF,
                              coeff_c,
                              coeff_blr,
                              pmt_active = pmt_active,
                              n_baseline = n_baseline)
    return deconv_pmt


def get_run_number(h5in):
    if "runInfo" in h5in.root.Run: return h5in.root.Run.runInfo[0]['run_number']
    else:                          return h5in.root.Run.RunInfo[0]['run_number']


class WfType(Enum):
    rwf  = 0
    mcrd = 1


def get_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf
    elif wf_type is WfType.mcrd: return h5in.root.    pmtrd ,   h5in.root.    sipmrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")


def wf_from_files(paths, wf_type):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            event_infos = h5in.root.Run.events
            run_number  = get_run_number(h5in)
            ( pmt_wfs,
             sipm_wfs) = get_wfs(h5in, wf_type)
            mc_tracks  = h5in.root.MC.MCTracks if run_number <= 0 else None
            for pmt, sipm, (event_number, timestamp) in zip(pmt_wfs, sipm_wfs, event_infos[:]):
                yield dict(pmt=pmt, sipm=sipm, mc=mc_tracks,
                           run_number=run_number, event_number=event_number, timestamp=timestamp)
            # NB, the monte_carlo writer is different from the others:
            # it needs to be given the WHOLE TABLE (rather than a
            # single event) at a time.


def sensor_data(path, wf_type):
    with tb.open_file(path, "r") as h5in:
        if   wf_type is WfType.rwf :   (pmt_wfs, sipm_wfs) = (h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf)
        elif wf_type is WfType.mcrd:   (pmt_wfs, sipm_wfs) = (h5in.root.    pmtrd ,   h5in.root.    sipmrd )
        else                       :   raise TypeError(f"Invalid WfType: {type(wf_type)}")
        _, NPMT ,  PMTWL =  pmt_wfs.shape
        _, NSIPM, SIPMWL = sipm_wfs.shape
        return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)
