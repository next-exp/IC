import os
import numpy  as np
import tables as tb

from .  zemrude        import zemrude
from .. types .symbols import NormMethod
from .. types .symbols import MapFitFunction
from .. types .symbols import SelRegionMethod
from .. types .symbols import EventRange

from .. core  .testing_utils import ignore_warning

from pytest import fixture

@fixture
def zemrude_config():
    run_number = 15504
    return dict(files_in    = os.path.expandvars(f'$ICDIR/database/test_data/run_{run_number}/*/*.h5'),
                file_out    = f'3Dmap_{run_number}.h5',
                compression = 'ZLIB4',
                event_range = EventRange.all,

                run_number = 15504,

                detector_db = 'next100',
                pre_map     = os.path.expandvars('$ICDIR/database/test_data/preliminary_map_15502.h5'),
                norm_method = NormMethod.maximum,

                # ------ Selection parameters ------
                dtrms2_low = lambda dt: -0.7 + 0.030 * (dt-20),
                dtrms2_upp = lambda dt: 2.6 + 0.036 * (dt-20),
                dtrms2_cen = lambda dt:  1.0 + 0.033 * (dt-20),
                low_xrays  = 36,
                high_xrays = 47,
                low_S2t    = 1.38e6,
                high_S2t   = 1.44e6,
                R_max      = 500,
                low_DT     = 20,
                high_DT    = 1350,
                low_nsipm  = 0,
                high_nsipm = 30,

                # ------ Create map parameters ------
                xy_range     = (-500, 500),
                dt_range     = (20, 1350),
                xy_nbins     = 1,
                dt_nbins     = 1,
                S2e_range    = (1000, 20000),
                fit_function = MapFitFunction.gaussian,
                min_events   = 40,
                nbins        = 100,

                # ------ Time evolution parameters ------
                slice_hours = 10000,
                x0          = 0,
                y0          = 0,
                shape       = SelRegionMethod.circle,
                shape_size  = 100,
                dtbins_dv   = np.linspace(1200, 1400),
                s1_DTrange  = (1000, 1350),
                bins_Ec     = np.linspace(20, 60, 101),
                error       = False,

                # ------ Make control plots parameters ------
                plots_out     = f'plots_out_{run_number}',
                ebins1        = np.linspace(0, 50, 101),
                ns1bins       = np.linspace(0, 10, 10),
                s1hbins       = np.linspace(0, 20, 100),
                s1wbins       = np.linspace(50, 1000, 20),
                ebins2        = np.arange(4e3, 1e4, 51),
                ns2bins       = np.linspace(0, 20, 20),
                s2hbins       = np.linspace(0, 3e3, 100),
                s2qbins       = np.linspace(0, 1.5e3, 100),
                qmaxbins      = np.linspace(0, 300, 100),
                s2wbins       = np.linspace(0, 60, 100),
                dtbins2       = np.linspace(0, 1400, 51),
                bins          = 100,
                dtr2_bins     = (20, 20),
                statistic     = 'mean',
                xy_range_plot = np.linspace(-500, 500, 100)
    )


@ignore_warning.str_length
@ignore_warning.no_config_group
def test_zemrude(zemrude_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'zemrude_output.h5')
    zemrude_config.update(dict(file_out = path_out))

    zemrude(**zemrude_config)

    nodes = ('krmap/krmap',
             't_evol/t_evol',
             'metadata',
             'data/selection_efficiencies')
    with tb.open_file(path_out) as h5out:
        for node in nodes:
            assert node in h5out.root
