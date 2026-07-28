"""
-----------------------------------------------------------------------
                              Penthesilea
-----------------------------------------------------------------------

From ancient Greek, Πενθεσίλεια: She, who brings suffering.

This city processes each S2 signal previously selected as pmaps in
irene assuming a unique S1 within an event to produce a set of
reconstructed energy depositions (hits). Hits consist of three
dimensional coordinates with associated energy (PMT signal) and charge
(SiPM signal). The city contains a peak/event filter, which can be
configured to find events with a certain number of S1/S2 signals that
satisfy certain properties. Currently, the city is designed to accept
only 1 S1 signal and will take the first S1 signal even if the filter
is configured to take more than 1 S1. Besides hits, the city also
stores the global (x, y) position of each S2 signal.
The tasks performed are:
    - Classify peaks according to the filter.
    - Filter out events that do not satisfy the selector conditions.
    - Rebin S2 signals.
    - Compute a set of hits for each slice in the rebinned S2 signal.
    - If there are more than one hit per slice, share the energy
      according to the charge recorded in the tracking plane.
"""
from operator import attrgetter

import tables as tb

from .. core.configure         import       EventRangeType
from .. core.configure         import       OneOrManyFiles
from .. core                   import       tbl_functions as tbl
from .. io  .          hits_io import          hits_writer
from .. io  . run_and_event_io import run_and_event_writer
from .. io  .          kdst_io import            kr_writer
from .. io  .  event_filter_io import  event_filter_writer

from .. dataflow          import dataflow as df
from .. dataflow.dataflow import     push
from .. dataflow.dataflow import     pipe
from .. types.symbols     import RebinMethod
from .. types.symbols     import  SiPMCharge
from .. types.symbols     import      XYReco

from .  components import                  city
from .  components import          copy_mc_info
from .  components import           print_every
from .  components import       peak_classifier
from .  components import   compute_xy_position
from .  components import       pmap_from_files
from .  components import           hit_builder
from .  components import               collect
from .  components import build_pointlike_event as build_pointlike_event_


def hitc_to_df(hitc: HitCollection):
    hits = []
    for hit in hitc.hits:
        hits.append(pd.DataFrame(dict( event    = hitc.event
                                     , time     = hitc.time
                                     , npeak    = hit .npeak
                                     , Xpeak    = hit .Xpeak
                                     , Ypeak    = hit .Ypeak
                                     , X        = hit .X
                                     , Y        = hit .Y
                                     , Z        = hit .Z
                                     , Q        = hit .Q
                                     , E        = hit .E
                                     , Ec       = hit .Ec
                                     , track_id = hit .track_id
                                     , Ep       = hit .Ep), index=[0]))
    df = pd.concat(hits, ignore_index=True)
    df = df.astype(dict(event=np.int64, npeak=np.uint16, Ec=np.float64, track_id=np.int64, Ep=np.float64))
    return df


@city
def penthesilea( files_in           : OneOrManyFiles
               , file_out           : str
               , compression        : str
               , event_range        : EventRangeType
               , print_mod          : int
               , detector_db        : str
               , run_number         : int
               , drift_v            : float
               , rebin              : int
               , s1_nmin            :   int, s1_nmax     :   int
               , s1_emin            : float, s1_emax     : float
               , s1_wmin            : float, s1_wmax     : float
               , s1_hmin            : float, s1_hmax     : float
               , s1_ethr            : float
               , s2_nmin            :   int, s2_nmax     :   int
               , s2_emin            : float, s2_emax     : float
               , s2_wmin            : float, s2_wmax     : float
               , s2_hmin            : float, s2_hmax     : float
               , s2_ethr            : float
               , s2_nsipmmin        :   int, s2_nsipmmax :   int
               , slice_reco_algo    : XYReco
               , global_reco_algo   : XYReco
               , slice_reco_params  : dict
               , global_reco_params : dict
               , rebin_method       : RebinMethod
               , sipm_charge_type   : SiPMCharge
               ):

    #  slice_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for hits reconstruction
    # global_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for overall global (pointlike event) reconstruction
    slice_reco  = compute_xy_position(detector_db, run_number,  slice_reco_algo, ** slice_reco_params)
    global_reco = compute_xy_position(detector_db, run_number, global_reco_algo, **global_reco_params)


    classify_peaks = df.map(peak_classifier(**locals()),
                            args = "pmap",
                            out  = "selector_output")

    pmap_passed           = df.map(attrgetter("passed"), args="selector_output", out="pmap_passed")
    pmap_select           = df.count_filter(bool, args="pmap_passed")

    build_hits            = df.map(hit_builder(detector_db, run_number, drift_v,
                                               rebin, rebin_method,
                                               global_reco, slice_reco,
                                               sipm_charge_type),
                                   args = ("pmap", "selector_output", "event_number", "timestamp"),
                                   out  = "hits"                                                 )

    to_hits_df            = df.map(hitc_to_df, item="hits")
    build_pointlike_event = df.map(build_pointlike_event_( detector_db, run_number, drift_v
                                                         , global_reco, sipm_charge_type),
                                   args = ("pmap", "selector_output", "event_number", "timestamp"),
                                   out  = "pointlike_event"                                      )

    event_count_in  = df.spy_count()
    event_count_out = df.spy_count()

    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        # Define writers...
        write_hits            = hits_writer(h5out, "RECO", "Events", compression=compression)
        write_hits            = df.sink(                 write_hits, args="hits")
        write_event_info      = df.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_pointlike_event = df.sink(           kr_writer(h5out), args="pointlike_event")
        write_pmap_filter     = df.sink( event_filter_writer(h5out, "s12_selector"), args=("event_number", "pmap_passed"))

        result = push(source = pmap_from_files(files_in),
                      pipe   = pipe(df.slice(*event_range, close_all=True)                ,
                                    print_every(print_mod)                                ,
                                    event_count_in.spy                                    ,
                                    classify_peaks                                        ,
                                    pmap_passed                                           ,
                                    df.branch(write_pmap_filter)                          ,
                                    pmap_select          .filter                          ,
                                    event_count_out      .spy                             ,
                                    df.branch("event_number", evtnum_collect.sink)        ,
                                    df.fork((build_hits, to_hits_df, write_hits           ),
                                            (build_pointlike_event, write_pointlike_event),
                                                                    write_event_info    )),
                      result = dict(events_in   = event_count_in .future,
                                    events_out  = event_count_out.future,
                                    evtnum_list = evtnum_collect .future,
                                    selection   = pmap_select    .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result



def hit_builder( detector_db : str
               , run_number  : int
               , drift_v     : float
               , rebin_method: RebinMethod
               , rebin_slices: Union[int, float]
               , global_reco : XYReco
               , slice_reco  : XYReco
               , charge_type : SiPMCharge
               ) -> Callable:
    """
    Builds hits from PMaps using a general clustering algorithm. For a given
    PMap, and the output of the peak-selector output does the following:
    - Filters out peaks rejected by the selector
    - Picks up the S1 (always the first one, if there are more, they are ignored)
    - Rebins each S2 according to `rebin_method` and `rebin_slices`
    - For each S2:
      - Compute the overall position of the signal according to `global_reco`
        (typically barycenter in XYZ)
      - For each (rebinned) slice of the S2:
        - Clusterize the SiPM responses according to `slice_reco`
          - Failing XY reconstructions (e.g. not enough SiPMs with signal)
            generate "empty" (a.k.a. NN) clusters
        - Assign each cluster the corresponding fraction of the energy in the
          slice

    Parameters
    ----------
    detector_db: str
      Detector database to use

    run_number: int
      Run number being processed

    drift_v: float
      Drift velocity in the data

    rebin_method: RebinMethod
      Which rebinning (resampling) algorithm to use

    rebin_slices: int or float
      Configuration option for `rebin_method`. It's interpretation depends on
      the method:
      If stride, `rebin_slices` represents the number of consecutive slices co
      merge into one.
      If threshold, `rebin_slices` represents the minimum charge a slice must
      have for it not to be rebinned.

    global_reco: Callable
      Reconstruction function to use for the event as a whole

    slice_reco: Callable
      Reconstruction function to use on each slice

    charge_type: SiPMCharge
      Interpretation of the SiPM charge.

    Returns
    -------
    build_hits: Callable
      A function that computes hits.
    """
    sipm_xys   = sipm_positions(detector_db, run_number)
    sipm_noise =   NoiseSampler(detector_db, run_number).signal_to_noise

    def build_hits( pmap           : PMap
                  , selector_output: S12SelectorOutput
                  , event_number   : int
                  , timestamp      : float
                  ) -> HitCollection:
        hitc = HitCollection(event_number, timestamp * 1e-3)
        s1_t = get_s1_time(pmap, selector_output)

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_method, rebin_slices)
            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)

            xy_peak     = try_global_reco(global_reco, xys, qs)
            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)

            slice_zs = (peak.times - s1_t) * units.ns * drift_v
            slice_es = peak.pmts.sum_over_sensors
            xys      = sipm_xys[peak.sipms.ids]

            for (z_slice, e_slice, sipm_qs) in zip(slice_zs, slice_es, sipm_charge):
                try:
                    clusters = slice_reco(xys, sipm_qs)
                    qs       = np.array([c.Q for c in clusters])
                    es       = hif.e_from_q(qs, e_slice)
                    for c, e in zip(clusters, es):
                        hit  = Hit(peak_no, c, z_slice, e, xy_peak)
                        hitc.hits.append(hit)
                except XYRecoFail:
                    hit = Hit(peak_no, Cluster.empty(), z_slice,
                              e_slice, xy_peak)
                    hitc.hits.append(hit)

        return hitc
    return build_hits
