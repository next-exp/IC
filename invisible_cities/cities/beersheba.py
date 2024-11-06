"""
-----------------------------------------------------------------------
                               Beersheba
-----------------------------------------------------------------------
Beersheba, a city suspended from the heavens, inhabited only by
idealists.

This city applies a Lucy-Rihcardson (LR) algorithm to reconstruct the
electron cloud density.

It reads hDSTs produced by Sophronia and produces deconvolved
hits. The LR deconvolution finds the (discretized) charge distribution
that generates the input image based on the PSF of the system. Each
bin in the charge distribution is interpreted as a (deconvolved)
hit.
The workflow of Beersheba can be summarized as:
  - Apply geometrical and lifetime corrections
  - Apply a low charge threshold to the input hits. If this leaves
    behind a slice with no hits, a fake hit (NN-hit) is temporarily
    created.
  - Merge NN-hits: The NN-hits' energy is reassigned to the closest
    non-NN-hits
  - Apply a second charge threshold with a slightly different energy
    redistribution
  - Drops isolated sensors (*)
  - Normalizes charge within each S2 peak
  - For each slice
    - Interpolates the hits to obtain a continuous image (real_image)
    - Generate a flat charge distribution as a seed
    - Do the following until a maximum number of iterations is reached
      or the difference between consecutive images is smaller than a
      threshold
      - Convolve the charge distribution with the PSF (new_image)
      - Update the charge distribution according to the difference
        between new_image and real_image
      - Clean up the image by applying a cut on the fraction of energy
        of the resulting charge-distribution hits
"""

import numpy  as np
import tables as tb
import pandas as pd

from os   .path  import expandvars
from scipy.stats import multivariate_normal
from numpy       import nan_to_num

from .  components import city
from .  components import collect
from .  components import copy_mc_info
from .  components import print_every
from .  components import hits_corrector
from .  components import hits_thresholder
from .  components import hits_and_kdst_from_files

from .. core.configure         import EventRangeType
from .. core.configure         import OneOrManyFiles
from .. core.configure         import check_annotations

from .. core                   import system_of_units as units
from .. core                   import tbl_functions   as tbl
from .. dataflow               import dataflow        as fl

from .. dataflow.dataflow      import push
from .. dataflow.dataflow      import pipe

from .. database.load_db       import DataSiPM

from .. reco.deconv_functions  import find_nearest
from .. reco.deconv_functions  import cut_and_redistribute_df
from .. reco.deconv_functions  import drop_isolated_sensors
from .. reco.deconv_functions  import deconvolve
from .. reco.deconv_functions  import richardson_lucy
from .. reco.deconv_functions  import no_satellite_killer

from .. io.run_and_event_io    import run_and_event_writer
from .. io.          dst_io    import df_writer
from .. io.          dst_io    import load_dst
from .. io.         hits_io    import hits_writer
from .. io. event_filter_io    import event_filter_writer
from .. io.         kdst_io    import kdst_from_df_writer

from .. types.ic_types         import NoneType
from .. types.symbols          import HitEnergy
from .. types.symbols          import InterpolationMethod
from .. types.symbols          import CutType
from .. types.symbols          import DeconvolutionMode


from typing import Tuple
from typing import List
from typing import Optional
from typing import Union


# Temporary. The removal of the event model will fix this.
def hitc_to_df_(hitc):
    columns = "event time npeak Xpeak Ypeak nsipm X Y Xrms Yrms Z Q E Qc Ec track_id Ep".split()
    columns = {col:[] for col in columns}

    for hit in hitc.hits:
        columns["event"   ].append(hitc.event)
        columns["time"    ].append(hitc.time)
        columns["npeak"   ].append(hit .npeak)
        columns["Xpeak"   ].append(hit .Xpeak)
        columns["Ypeak"   ].append(hit .Ypeak)
        columns["nsipm"   ].append(hit .nsipm)
        columns["X"       ].append(hit .X)
        columns["Y"       ].append(hit .Y)
        columns["Xrms"    ].append(hit .Xrms)
        columns["Yrms"    ].append(hit .Yrms)
        columns["Z"       ].append(hit .Z)
        columns["Q"       ].append(hit .Q)
        columns["E"       ].append(hit .E)
        columns["Qc"      ].append(hit .Qc)
        columns["Ec"      ].append(hit .Ec)
        columns["track_id"].append(hit .track_id)
        columns["Ep"      ].append(hit .Ep)
    return pd.DataFrame(columns)

def event_info_adder(timestamp : float, dst : pd.DataFrame):
    return dst.assign(time=timestamp/1e3, nsipm=0, Xrms=0, Yrms=0)


@check_annotations
def deconvolve_signal(det_db           : pd.DataFrame,
                      psf_fname        : str,
                      e_cut            : float,
                      n_iterations     : int,
                      iteration_tol    : float,
                      sample_width     : List[float],
                      bin_size         : List[float],
                      satellite_params : Union[dict, NoneType],
                      diffusion        : Optional[Tuple[float, float, float]]=(1., 1., 0.3),
                      energy_type      : Optional[HitEnergy]=HitEnergy.Ec,
                      deconv_mode      : Optional[DeconvolutionMode]=DeconvolutionMode.joint,
                      n_dim            : Optional[int]=2,
                      cut_type         : Optional[CutType]=CutType.abs,
                      inter_method     : Optional[InterpolationMethod]=InterpolationMethod.cubic,
                      n_iterations_g   : Optional[int]=0):
    """
    Applies Lucy Richardson deconvolution to SiPM response with a
    given set of PSFs and parameters.

    Parameters
    ----------
    det_db           : Detector database.
    psf_fname        : Point-spread function.
    e_cut            : Cut in absolute/relative value to the max voxel over the deconvolution output.
    n_iterations     : Number of Lucy-Richardson iterations
    iteration_tol    : Stopping threshold (difference between iterations).
    sample_width     : Sampling size of the sensors.
    bin_size         : Size of the interpolated bins.
    satellite_params : Dictionary containing parameters for satellite killer
        satellite_start_iter : Iteration no. when satellite killer starts being used.
        satellite_max_size   : Maximum size of satellite deposit, above which they are considered 'real'.
        e_cut                : Cut in absolute/relative value to the provided
                               deconvolution output for satellite discrimination.
        cut_type             : Cut mode within satellite killer, see definition of `cut_type` below.
    energy_type      : Energy type (`E` or `Ec`, see Esmeralda) used for assignment.
    deconv_mode      : `joint` or `separate`, 1 or 2 step deconvolution, see description later.
    diffusion        : Diffusion coefficients in each dimension for 'separate' mode.
    n_dim            : Number of dimensions to apply the method (usually 2).
    cut_type         : Cut mode to the deconvolution output (`abs` or `rel`) using e_cut
                        `abs`: cut on the absolute value of the hits.
                        `rel`: cut on the relative value (to the max) of the hits.
    inter_method     : Interpolation method (`nointerpolation`, `nearest`, `linear` or `cubic`).
    n_iterations_g   : Number of Lucy-Richardson iterations for gaussian in 'separate mode'

    Returns
    ----------
    apply_deconvolution : Function that takes hits and returns the
    deconvolved data.
    """
    if satellite_params is None:
        satellite_params = no_satellite_killer

    dimensions    = np.array  (['X', 'Y', 'Z'][:n_dim])
    bin_size      = np.asarray(bin_size               )
    diffusion     = np.asarray(diffusion              )

    psfs          = load_dst(psf_fname, 'PSF', 'PSFs')
    det_grid      = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
                     for var, bs in zip(dimensions, bin_size)]
    deconvolution = deconvolve(n_iterations, iteration_tol,
                               sample_width, det_grid,
                               **satellite_params,
                               inter_method = inter_method)

    if not isinstance(energy_type , HitEnergy          ):
        raise ValueError(f'energy_type {energy_type} is not a valid energy type.')
    if not isinstance(inter_method, InterpolationMethod):
        raise ValueError(f'inter_method {inter_method} is not a valid interpolation method.')
    if not isinstance(cut_type    , CutType            ):
        raise ValueError(f'cut_type {cut_type} is not a valid cut type.')
    if not isinstance(deconv_mode , DeconvolutionMode  ):
        raise ValueError(f'deconv_mode {deconv_mode} is not a valid deconvolution mode.')

    def deconvolve_hits(df, z):
        '''
        Given an slice, applies deconvolution using the PSF
        associated to the passed z.

        Parameters
        ----------
        df : Original input dataframe for the deconvolution (single slice cdst)
        z  : Longitudinal position of the slice.
        Returns
        ----------
        Dataframe with the deconvolved slice.
        '''
        xx, yy = df.Xpeak.unique(), df.Ypeak.unique()
        zz     = z if deconv_mode is DeconvolutionMode.joint else 0
        psf = psfs.loc[(psfs.z == find_nearest(psfs.z, zz)) &
                       (psfs.x == find_nearest(psfs.x, xx)) &
                       (psfs.y == find_nearest(psfs.y, yy)) , :]

        deconv_image, pos = deconvolution(tuple(df.loc[:, dimensions].values.T), df.NormQ.values, psf)

        if   deconv_mode is DeconvolutionMode.joint:
            pass
        elif deconv_mode is DeconvolutionMode.separate:
            dist         = multivariate_normal(np.zeros(n_dim), diffusion**2 * z * units.mm / units.cm) #Z is in mm in cdst
            cols         = tuple(f"{v.lower()}r" for v in dimensions)
            psf_cols     = psf.loc[:, cols]
            gaus         = dist.pdf(psf_cols.values)
            psf          = gaus.reshape(psf_cols.nunique())
            deconv_image = nan_to_num(richardson_lucy(deconv_image, psf,
                                                      iterations = n_iterations_g,
                                                      iter_thr = iteration_tol,
                                                      **satellite_params))

        return create_deconvolution_df(df, deconv_image.flatten(), pos, cut_type, e_cut, n_dim)

    def apply_deconvolution(df):
        '''
        Given an event cdst, it iterates through its S2s and applies deconvolution
        to each S2.

        Parameters
        ----------
        df : Original input dataframe for the deconvolution (event cdst)

        Returns
        ----------
        Dataframe with the deconvolved event.
        '''
        deco_dst = []
        df.loc[:, "NormQ"] = np.nan
        for peak, hits in df.groupby("npeak"):
            hits.loc[:, "NormQ"] = hits.loc[:, 'Q'] / hits.loc[:, 'Q'].sum()
            deconvolved_hits = pd.concat([deconvolve_hits(df_z, z) for z, df_z in hits.groupby("Z")], ignore_index=True)
            deconvolved_hits = deconvolved_hits.assign(npeak=peak, Xpeak=hits.Xpeak.iloc[0], Ypeak=hits.Ypeak.iloc[0])
            distribute_energy(deconvolved_hits, hits, energy_type)
            deco_dst.append(deconvolved_hits)

        return pd.concat(deco_dst, ignore_index=True)

    return apply_deconvolution


def create_deconvolution_df(hits, deconv_e, pos, cut_type, e_cut, n_dim):
    '''
    Given the output of the deconvolution, it cuts the low energy voxels and
    creates a dataframe object with the resulting output.

    Parameters
    ----------
    hits     : Original input dataframe for the deconvolution (S2 cdst)
    deconv_e : Deconvolution energy distribution (n-dim array)
    pos      : Position of the deconvolved hits.
    cut_type : CutType object with the cut mode.
    e_cut    : Value for the energy cut.
    n_dim    : Number of dimensions of the deconvolution (tipically 2 as of now)

    Returns
    ----------
    df       : Dataframe with the deconvolution input after energy cutting.
    '''

    df  = pd.DataFrame(columns=['event', 'npeak', 'X', 'Y', 'Z', 'E'])

    if   cut_type is CutType.abs:
        sel_deconv = deconv_e > e_cut
    elif cut_type is CutType.rel:
        sel_deconv = deconv_e / deconv_e.max() > e_cut
    else:
        raise ValueError(f'cut_type {cut_type} is not a valid cut type.')

    df['E']     = deconv_e[sel_deconv]
    df['event'] = hits.event.unique()[0]
    df['npeak'] = hits.npeak.unique()[0]
    df['Z']     = hits.Z    .unique()[0] if n_dim == 2 else pos[2][sel_deconv]
    df['X']     = pos[0][sel_deconv]
    df['Y']     = pos[1][sel_deconv]

    return df


def distribute_energy(df, cdst, energy_type):
    '''
    Assign the energy of a dataframe (cdst) to another dataframe (deconvolved),
    distributing it according to the charge fraction of each deconvolution hit.

    Parameters
    ----------
    df          : Deconvolved dataframe with a single S2 (npeak)
    cdst        : Dataframe with the sensor response (usually a cdst)
    energy_type : HitEnergy with which 'type' of energy should be assigned.
    '''
    df.loc[:, 'E'] = df.E / df.E.sum() * cdst.loc[:, energy_type.value].sum()


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

    def cut_over_Q(df):  # df shall be an event cdst
        cdst = df.groupby(['event', 'npeak']).apply(cut).reset_index(drop=True)

        return cdst

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

    def drop_isolated(df): # df shall be an event cdst
        df = df.groupby(['event', 'npeak']).apply(drop).reset_index(drop=True)

        return df

    return drop_isolated


def check_nonempty_dataframe(df) -> bool:
    """
    Filter for Beersheba flow. The flow stops if:
        1. there are no hits (after droping isolated sensors)
    """
    return len(df) > 0


def deconv_writer(h5out):
    """
    For a given open table returns a writer for deconvolution hits dataframe
    """
    def write_deconv(df):
        return df_writer(h5out              = h5out             ,
                         df                 = df                ,
                         group_name         = 'DECO'            ,
                         table_name         = 'Events'          ,
                         descriptive_string = 'Deconvolved hits',
                         columns_to_index   = ['event']         )
    return write_deconv



@city
def beersheba( files_in         : OneOrManyFiles
             , file_out         : str
             , compression      : str
             , event_range      : EventRangeType
             , print_mod        : int
             , detector_db      : str
             , run_number       : int
             , threshold        : float
             , same_peak        : bool
             , deconv_params    : dict
             , satellite_params : Union[dict, NoneType]
             , corrections      : dict
             ):
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

    threshold     : float
        Threshold to be applied to all SiPMs.
    same_peak     : bool
        Whether to reassign NN hits within the same peak.
    deconv_params : dict
        q_cut                : float
            Minimum charge (pes) on a hit (SiPM)
        drop_dist            : float
            Distance to check if a SiPM is isolated
        psf_fname            : string (filepath)
            Filename of the psf
        e_cut                : float
            Cut over the deconvolution output, arbitrary units (order 1e-3)
        n_iterations         : int
            Number of iterations to be applied if the iteration_tol criteria
            is not fulfilled before.
        iteration_tol        : float
            Stopping threshold (difference between iterations). I
        sample_width         : list[float]
            Sampling of the sensors in each dimension (usuallly the pitch).
        bin_size             : list[float]
            Bin size (mm) of the deconvolved image.
        energy_type          : HitEnergy (`E` or `Ec`)
            Marks which energy from Esmeralda (E = uncorrected, Ec = corrected)
            should be assigned to the deconvolved track.
        deconv_mode          : DeconvolutionMode (`joint` or `separate`)
            - joint deconvolves once using a PSF based on Z that includes
               both EL and diffusion spread aproximated to a Z range.
            - separate deconvolves twice, first using the EL PSF, then using
               a gaussian PSF based on the exact Z position of the slice.
        diffusion            : tuple(float)
            Diffusion coefficients in each dimmension (mm/sqrt(cm))
            used if deconv_mode is `separate`
        n_dim                : int
            Number of dimensions used in deconvolution, currently only 2 max:
            n_dim = 2 -> slice by slice XY deconvolution.
            n_dim = 3 -> XYZ deconvolution (in the works).
        inter_method         : InterpolationMethod (`nointerpolation`, `nearest`, `linear` or `cubic`)
            Sensor interpolation method. If None, no interpolation will be applied.
            'cubic' not supported for 3D deconvolution.
        n_iterations_g       : int
            Number of Lucy-Richardson iterations for gaussian in 'separate mode'
    satellite_params : dict, None
        satellite_start_iter : int
            Iteration no. when satellite killer starts being used.
        satellite_max_size   : int
            Maximum size of satellite deposit, above which they are considered 'real'.
        e_cut                : float
            Cut over the deconvolution input, for relevant satellite discrimination
        cut_type             : CutType
            Cut mode to the deconvolution output (`abs` or `rel`) using e_cut
              `abs`: cut on the absolute value of the hits.
              `rel`: cut on the relative value (to the max) of the hits.

    corrections : dict
        filename : str
            Path to the file holding the correction maps
        apply_temp : bool
            Whether to apply temporal corrections
        norm_strat : NormStrategy
            Normalization strategy
        norm_value : float, optional
            Normalization value in case of `norm_strat = NormStrategy.custom`

    ----------
    Input
    ----------
    Esmeralda output
    ----------
    Output
    ----------
    DECO    : Deconvolved hits table
    MC info : (if run number <=0)
    """
    correct_hits   = fl.map(hits_corrector(**corrections), item="hits")
    threshold_hits = fl.map(hits_thresholder(threshold, same_peak), item="hits")
    hitc_to_df     = fl.map(hitc_to_df_, item="hits")

    deconv_params['psf_fname'       ] = expandvars(deconv_params['psf_fname'])
    deconv_params['satellite_params'] = satellite_params

    for p in ['sample_width', 'bin_size', 'diffusion']:
        if len(deconv_params[p]) != deconv_params['n_dim']:
            raise ValueError         (f"Parameter {p} dimensions do not match n_dim parameter")
    if deconv_params['n_dim'] > 2:
        raise     NotImplementedError(f"{deconv_params['n_dim']}-dimensional PSF not yet implemented")

    cut_sensors           = fl.map(cut_over_Q   (deconv_params.pop("q_cut")    , ['E', 'Ec']),
                                   item = 'hits')
    drop_sensors          = fl.map(drop_isolated(deconv_params.pop("drop_dist"), ['E', 'Ec']),
                                   item = 'hits')
    filter_events_no_hits = fl.map(check_nonempty_dataframe,
                                   args = 'hits',
                                   out  = 'hits_passed_no_hits')
    deconvolve_events     = fl.map(deconvolve_signal(DataSiPM(detector_db, run_number), **deconv_params),
                                   args = 'hits',
                                   out  = 'deconv_dst')

    add_event_info        = fl.map(event_info_adder, args=("timestamp", "deconv_dst"), out="deconv_dst")

    event_count_in        = fl.spy_count()
    event_count_out       = fl.spy_count()
    events_passed_no_hits = fl.count_filter(bool, args = "hits_passed_no_hits")

    filter_out_none       = fl.filter(lambda x: x is not None, args = "kdst")

    evtnum_collect        = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        # Define writers
        write_event_info    = fl.sink(run_and_event_writer(h5out), args = ("run_number", "event_number", "timestamp"))
        write_deconv        = fl.sink(       deconv_writer(h5out), args =  "deconv_dst")
        write_kdst_table    = fl.sink( kdst_from_df_writer(h5out), args =  "kdst"      )
        write_thr_hits      = fl.sink(         hits_writer(h5out, "CHITS", "lowTh"), args = "hits")
        write_nohits_filter = fl.sink( event_filter_writer(h5out, "nohits"), args=("event_number", "hits_passed_no_hits"))

        result = push(source = hits_and_kdst_from_files(files_in, "RECO", "Events"),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)    ,
                                    print_every(print_mod)                    ,
                                    event_count_in.spy                        ,
                                    correct_hits                              ,
                                    threshold_hits                            ,
                                    fl.branch(write_thr_hits)                 ,
                                    hitc_to_df                                ,
                                    cut_sensors                               ,
                                    drop_sensors                              ,
                                    filter_events_no_hits                     ,
                                    fl.branch(write_nohits_filter)            ,
                                    events_passed_no_hits.filter              ,
                                    deconvolve_events                         ,
                                    add_event_info                            ,
                                    event_count_out.spy                       ,
                                    fl.branch("event_number"     ,
                                              evtnum_collect.sink)            ,
                                    fl.fork(write_deconv    ,
                                            (filter_out_none, write_kdst_table),
                                            write_event_info))                ,
                      result = dict(events_in   = event_count_in       .future,
                                    events_out  = event_count_out      .future,
                                    evtnum_list = evtnum_collect       .future,
                                    events_pass = events_passed_no_hits.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
