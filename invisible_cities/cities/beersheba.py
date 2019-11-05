"""
-----------------------------------------------------------------------
                              Beersheba
-----------------------------------------------------------------------
Beersheba, a city suspended from the heavens, inhabited only by idealists.
This city interpolates corrected hits and applies Lucy-Richardson deconvolution
to the interpolated signal.
The input is esmeralda output containing hits, kdst global information and mc info.
The city outputs :
    - DECO deconvolved hits table
    - MC info (if run number <=0)
    - SUMMARY summary of per event information
"""

import tables as tb
import pandas as pd

from scipy.stats import multivariate_normal
from numpy       import sqrt
from numpy       import nan_to_num


from typing      import Tuple
from typing      import List
from typing      import Callable
from typing      import Optional

from .  components import city
from .  components import print_every
from .  components import hdst_from_files

from .. reco                  import tbl_functions           as tbl
from .. dataflow              import dataflow                as fl

from .. dataflow.dataflow     import push
from .. dataflow.dataflow     import pipe

from .. reco.deconv_functions import find_nearest
from .. reco.deconv_functions import cut_and_redistribute_df
from .. reco.deconv_functions import drop_isolated_sensors
from .. reco.deconv_functions import deconvolve
from .. reco.deconv_functions import richardson_lucy

from .. core.core_functions   import weighted_mean_and_std

from .. io.       mcinfo_io   import mc_info_writer
from .. io.run_and_event_io   import run_and_event_writer
from .. io.          dst_io   import _store_pandas_as_tables

def deconvolve_signal(psf_fname       : str,
                      ecut            : float,
                      iterationNumber : int,
                      iterationThr    : float,
                      sampleWidth     : List[float],
                      bin_size        : List[float],
                      diffusion       : Tuple[float]=(1., 1., 0.3),
                      iterationGauss  : int=0,
                      energy_type     : str='Ec',
                      deconv_mode     : str='joint',
                      n_dim           : int=2,
                      cut_type        : str='abs',
                      interMethod     : str=None):

    """
    Applies Lucy Richardson deconvolution to SiPM response with a
    given set of PSFs and parameters.

    Parameters
    ----------
    psf_fname       : Point-spread function.
    ecut            : Cut in relative value to the max voxel over the deconvolution output.
    iterationNumber : Number of Lucy-Richardson iterations
    iterationGauss  : Number of Lucy-Richardson iterations for gaussian in 'separate mode'
    iterationThr    : Stopping threshold (difference between iterations).
    sampleWidth     : Sampling size of the sensors.
    bin_size        : Size of the interpolated bins.
    energy_type     : Energy type ('E' or 'Ec', see Esmeralda) used for assignment.
    deconv_mode     : 'joint' or 'separate', 1 or 2 step deconvolution, see description later.
    diffusion       : Diffusion coefficients in each dimension for 'separate' mode.
    n_dim           : Number of dimensions to apply the method (2 or 3).
    cut_type        : Cut mode to the deconvolution output ('abs' or 'rel') using ecut
                      'abs': cut on the absolute value of the hits.
                      'rel': cut on the relative value (to the max) of the hits.
    interMethod     : Interpolation method.

    Returns
    ----------
    apply_deconvolution : Function that takes hits and returns the
    deconvolved data.
    """

    psfs          = pd.read_hdf(psf_fname)
    deconvolution = deconvolve(iterationNumber, iterationThr, sampleWidth[:n_dim], bin_size[:n_dim], interMethod)
    dimensions    = ['X', 'Y', 'Z'][:n_dim]

    def create_deconvolution_df(hits, deconvE, pos):
        df = pd.DataFrame(columns=['event', 'npeak', 'X', 'Y', 'Z', 'E'])
        if   cut_type == 'abs':
            selDeconv   = deconvE > ecut
        elif cut_type == 'rel':
            selDeconv   = deconvE/deconvE.max() > ecut
        else:
            raise TypeError(f'Cut type "{cut_type}" unsupported')
        ene         = deconvE[selDeconv]
        df['event'] = [hits.event.unique()[0]] * len(ene)
        df['npeak'] = [hits.npeak.unique()[0]] * len(ene)
        df['Z']     = [hits.Z    .unique()[0]] * len(ene) if n_dim == 2 else pos[2][selDeconv]
        df['X']     = pos[0][selDeconv]
        df['Y']     = pos[1][selDeconv]
        df['E']     = ene

        return df

    def distribute_energy(df, hdst):
        df['E'] = df['E'] / df['E'].sum() * hdst[energy_type].sum()

        return df

    def deconvolve_hits(df, z):
        if   deconv_mode == 'joint':
            psf = psfs[(psfs.z == find_nearest(psfs.z,                 z)) &
                       (psfs.x == find_nearest(psfs.x, df.Xpeak.unique())) &
                       (psfs.y == find_nearest(psfs.y, df.Ypeak.unique())) ]
            deconvImage, pos = deconvolution(tuple(df[v].values for v in dimensions), df.Q.values, psf)
        elif deconv_mode == 'separate':
            psf_z0 = psfs[(psfs.z == find_nearest(psfs.z,                 0)) &
                          (psfs.x == find_nearest(psfs.x, df.Xpeak.unique())) &
                          (psfs.y == find_nearest(psfs.y, df.Ypeak.unique())) ]
            deconvImage, pos = deconvolution(tuple(df[v].values for v in dimensions), df.Q.values, psf_z0)
            g   = multivariate_normal(tuple(0. for _ in range(n_dim)), tuple((diff * sqrt(z/10))**2 for diff in diffusion[:n_dim])).pdf(list(zip(*tuple(psf_z0[f'{v.lower()}r'].values for v in dimensions))))
            psf = g.reshape(tuple(psf_z0[f'{v.lower()}r'].nunique() for v in dimensions))
            deconvImage      = nan_to_num(richardson_lucy(deconvImage, psf, iterationGauss, iterationThr))
        else:
            raise TypeError(f'Mode "{deconv_mode}" unsupported')

        return create_deconvolution_df(df, deconvImage.flatten(), pos)

    def apply_deconvolution(df):
        deco_dst = []
        for peak in sorted(set(df.npeak)):
            deco_df = []
            hits          = df[df.npeak == peak]
            hits._is_copy = False
            hits['Q']     = hits.Q/hits.Q.sum()
            deconvolved_hits = (deconvolve_hits(hits, weighted_mean_and_std(hits.Z, hits.E)[0]) if n_dim == 3 else
                                pd.concat([deconvolve_hits(hits[hits.Z == z], z) for z in hits.Z.unique()], ignore_index=True))
            deco_dst.append(distribute_energy(deconvolved_hits, hits))

        return pd.concat(deco_dst, ignore_index=True)

    return apply_deconvolution


def cut_over_Q(q_cut, redist_var):
    '''
    Apply a cut over the SiPM charge condition to hits and redistribute the
    energy variables.

    Parameters
    ----------
    q_cut      : Charge value over which to cut.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    cut_over_Q : Function that will cut the dataframe and redistribute
    values.
    '''
    cut = cut_and_redistribute_df(f"Q > {q_cut}", redist_var)

    def cut_over_Q(df):
        hdst = df.groupby(['event', 'npeak']).apply(cut).reset_index(drop=True)

        return hdst

    return cut_over_Q


def drop_isolated(distance, redist_var):
    """
    Drops rogue/isolated hits (SiPMs) from hits.

    Parameters
    ----------
    distance   : Sensor pitch.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    drop_isolated_sensors : Function that will drop the isolated sensors.
    """
    drop = drop_isolated_sensors(distance, redist_var)

    def drop_isolated(df):
        df = df.groupby(['event', 'npeak']).apply(drop).reset_index(drop=True)

        return df

    return drop_isolated


def deconv_writer(h5out, compression='ZLIB4', group_name='DECO', table_name='Events', descriptive_string='Deconvolved hits', str_col_length=32):
    """
    For a given open table returns a writer for deconvolution hits dataframe
    """
    def write_deconv(df):
        return _store_pandas_as_tables(h5out = h5out, df = df, compression = compression, group_name = group_name, table_name = table_name, descriptive_string = descriptive_string, str_col_length = str_col_length)
    return write_deconv


def summary_writer(h5out, compression='ZLIB4', group_name='SUMMARY', table_name='Events', descriptive_string='Event summary information', str_col_length=32):
    """
    For a given open table returns a writer for summary info dataframe
    """
    def write_summary(df):
        return _store_pandas_as_tables(h5out = h5out, df = df, compression = compression, group_name = group_name, table_name = table_name, descriptive_string = descriptive_string, str_col_length = str_col_length)
    return write_summary


@city
def beersheba(files_in, file_out, compression, event_range, print_mod, run_number,
              deconv_params = dict()):
    """
    The city corrects Penthesilea hits energy and extracts topology information.
    ----------
    Parameters
    ----------
    files_in    : str, filepath
         Input file
    file_out    : str, filepath
         Output file
    compression : str
         Default  'ZLIB4'
    event_range : int /'all_events'
         Number of events from files_in to process
    print_mode  : int
         How frequently to print events
    run_number  : int
         Has to be negative for MC runs

    deconv_params : dict
        q_cut           : float
            Minimum charge (pes) on a hit (SiPM)
        drop_dist       : float
            Distance to check if a SiPM is isolated
        psf_fname       : string (filepath)
            Filename of the psf
        ecut            : float
            Cut over the deconvolution output, arbitrary units (order 1e-3)
        iterationNumber : int
            Number of iterations to be applied if the iterationThr criteria
            is not fulfilled before.
        iterationThr    : float
            Stopping threshold (difference between iterations). I
        sampleWidth     : list[float]
            Sampling of the sensors in each dimension (usuallly the pitch).
        bin_size        : list[float]
            Bin size (mm) of the deconvolved image.
        energy_type     : str ('E', 'Ec')
            Marks which energy from Esmeralda (E = uncorrected, Ec = corrected)
            should be assigned to the deconvolved track.
        deconv_mode     : str ('joint', 'separate')
            - 'joint' deconvolves once using a PSF based on Z that includes
               both EL and diffusion spread aproximated to a Z range.
            - 'separate' deconvolves twice, first using the EL PSF, then using
               a gaussian PSF based on the exact Z position of the slice.
        diffusion       : tuple(float)
            Diffusion coefficients in each dimmension (mm/sqrt(cm))
            used if deconv_mode is 'separate'
        n_dim           : int
            Number of dimensions used in deconvolution, either 2 or 3:
            n_dim = 2 -> slice by slice XY deconvolution.
            n_dim = 3 -> XYZ deconvolution.
        interMethod     : str (None, 'linear', 'cubic')
            Sensor interpolation method. If None, no interpolation will be applied.
            'cubic' not supported for 3D deconvolution.

    ----------
    Input
    ----------
    Esmeralda output
    ----------
    Output
    ----------
    DECO    : Deconvolved hits table
    MC info : (if run number <=0)
    SUMMARY : Table with the summary from Esmeralda.
"""

    deconv_params_   = {k : v for k, v in deconv_params.items() if k not in ['q_cut', 'drop_dist']}

    cut_sensors       = fl.map(cut_over_Q   (deconv_params['q_cut'    ], ['E', 'Ec']),
                               args = 'hdst',
                               out  = 'hdst_cut')

    drop_sensors      = fl.map(drop_isolated(deconv_params['drop_dist'], ['E', 'Ec']),
                               args = 'hdst_cut',
                               out  = 'hdst_drop')

    deconvolve_events = fl.map(deconvolve_signal(**deconv_params_),
                               args = 'hdst_drop',
                               out  = 'deconv_dst')

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_mc_        = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_mc             = fl.sink(                   write_mc_, args = ("mc", "event_number"))
        write_deconv         = fl.sink(  deconv_writer(h5out=h5out), args =  "deconv_dst"         )
        write_summary        = fl.sink( summary_writer(h5out=h5out), args =  "summary"            )
        return push(source = hdst_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        event_count_in           .spy         ,
                        cut_sensors                           ,
                        drop_sensors                          ,
                        deconvolve_events                     ,
                        event_count_out          .spy         ,
                        fl.fork(write_mc        ,
                                write_deconv    ,
                                write_summary   ,
                                write_event_info))                 ,
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future))
