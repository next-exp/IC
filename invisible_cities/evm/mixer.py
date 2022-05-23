import os
import glob
import warnings
import tables as tb
import pandas as pd
import numpy  as np

from ..io.dst_io import load_dst
from ..io.dst_io import df_writer

from ..core.system_of_units import mBq
from ..database.load_db     import RadioactivityData

get_file_number = lambda filename: int(filename.split("/")[-1].split(".h5")[0].split("_")[1])


class Event_Mixer():
    '''
    This class writes MC mixed isaura files. It reads the files separately for each
    MC component pair (isotope, g4volume), selects the provided number of events
    for each component, and writes them consecutively.
    The input files and events are read randomly and therefore the output is non-deterministic.

    Parameters:
    ----------
    :inpath: (str) the input path of the MC files, which must explicitly depend on the simulated component
             through "g4volume" and "isotope" variables. For example: "/somepath/{g4volume}/{isotope}"

    :outpath:(str) the output path for where mixed files will be saved.

    :events_df: (pd.DataFrame) dataframe with 3 columns: G4Volume, Isotope, nevts

    :nevents_per_file: (int) the number of events to same in each mixed file. Each mixed file will
                      contain the fraction of events for each component based on :events_df:
    '''

    def __init__(self, inpath   : str
                     , outpath  : str
                     , nevents_df: pd.DataFrame
                     , nevents_per_file: int
                     , verbosity: int = 0):
        '''
        Initializes the constant (fixed-value) class atributes, except for self.counter
        which is modified some of the methods.
        '''

        self.verbosity = verbosity
        if self.verbosity: print(">> Initializing mixer...", end="\n")
        self.inpath  = os.path.expandvars(inpath)
        self.outpath = os.path.expandvars(outpath)

        # cast and set index
        self.nevents_df = nevents_df.astype({"nevts": int}, copy=True)
        self.nevents_df = self.nevents_df.set_index(["Isotope", "G4Volume"])
        self.nevents_df.to_csv(os.path.dirname(self.outpath) + "/nevents.csv")

        self.nevents_per_file = nevents_per_file
        self.saved_events = 0

        if self.verbosity:
            print(" ------------------------", end="\n")
            print(" Number of events to mix:", end="\n")
            print(" ------------------------", end="\n")
            print( self.nevents_df.__repr__(), end="\n")
            print("Total:", self.nevents_df.nevts.sum(), end="\n")

        # number of output files
        self.nfiles_out = \
            int(np.ceil(self.nevents_df.nevts.sum() / self.nevents_per_file))

        # (group, node) pair of isaura file tables (ignoring Filters)
        self.tables = dict(( (      "events", ("Run", "events")  )
                           , (    "eventMap", ("Run", "eventMap"))
                           , (         "dst", ("DST", "Events")  )
                           , (     "summary", ("Summary","Events"))
                           , (    "tracking", ("Tracking", "Tracks"))
                           , (        "hits", ("MC", "hits")    )
                           , (   "particles", ("MC", "particles"))
                           , ("sns_response", ("MC", "sns_response"))
                           ))

        # output file
        self.out_file_number = 0
        self.h5out = tb.open_file(self.outpath.format(file_number=0), "w")
        self.nevents_in_file = 0
        if self.verbosity:
            print(">> Writing:", self.outpath.format(file_number=self.out_file_number), end="\n")
        return


    def run(self):
        '''
        Runs the event mixer.
        For each (isotope, g4volume) component:
        - shufle input filenames
        - read and write data until all required events are saved
        Once all the output files are writen, the working dataframes are disposed.
        '''

        for isotope, g4volume in self.nevents_df.index:

            total_nevts = self.nevents_df.loc[isotope, g4volume].nevts
            if total_nevts == 0: continue
            filenames = sorted( glob.glob(self.inpath.format(g4volume = g4volume, isotope = isotope))
                              , key = get_file_number)

            np.random.shuffle(filenames)

            written_nevents = 0
            for filename in filenames:

                self.events_ = load_dst(filename, *self.tables["events"])
                if len(self.events_) == 0: continue
                self._read_data(filename, isotope, g4volume)

                # write nevts to output files
                nevts = min(total_nevts-written_nevents, len(self.events_))
                self._write_data(nevts)
                written_nevents += nevts

                if (written_nevents == total_nevts): break
        self.h5out.close()
        self._dispose_dfs()

        # check total number of events
        assert (self.saved_events == self.nevents_df.nevts.sum())
        return


    def _read_data(self, filename, isotope, g4volume):
        '''
        Reads data for each table in filename: self.name = load_dst(filename, group, node)
        Notice that index is set to the event number, simplifying data selection at self._select_data
        Also drops timestamp and adds file_number and component (isotope, g4volume) in dataframe columns
        '''

        for key, table in self.tables.items():

            if (key == "eventMap"):
                setattr(self, key + "_", load_dst(filename, *table).set_index("evt_number"))

            elif (key in ("dst", "tracking", "summary")):
                setattr(self, key + "_", load_dst(filename, *table)
                        .rename({"event": "evt_number"}, axis=1).set_index("evt_number"))

            elif (key in ("hits", "particles", "sns_response")):
                setattr(self, key + "_", load_dst(filename, *table).set_index("event_id"))

        # drop timestamp
        self.events_.drop("timestamp", axis=1, inplace=True)
        self.dst_   .drop(     "time", axis=1, inplace=True)

        # add file number and component info
        file_number = get_file_number(filename)
        for key, table in self.tables.items():
            if (key in ("events", "eventMap", "dst", "tracking", "summary")):
                exec(f"self.{key}_.loc[:,     'file'] = {file_number}")
                exec(f"self.{key}_.loc[:,  'Isotope'] = '{isotope}'")
                exec(f"self.{key}_.loc[:, 'G4Volume'] = '{g4volume}'")
        return


    def _write_data(self, nevents):
        '''
        Write nevents from current component dataframes to output files:
            - randomly selects nevents from dataframes
            - if current file has enough space, save the data there
            - if the space is not enough, fill the current file and continue writing
              in new files.
        '''

        # select nevents
        self.events_ = self.events_.sample(n=nevents)

        # write data to output file
        # empty space in current output file
        nidle = int(self.nevents_per_file - self.nevents_in_file)
        if (nidle >= nevents):
            self.events = self.events_
            self._select_data()

            df_writer(self.h5out, self.events, "Run", "events", 'ZLIB4')
            for key, table in self.tables.items():
                if key != "events":
                    exec(f"df_writer(self.h5out, self.{key}.reset_index(), *table, 'ZLIB4')")
            self.nevents_in_file += nevents

            if (nidle == nevents): # open new file
                self.h5out.close()
                self.out_file_number += 1
                self.nevents_in_file  = 0
                self.h5out = tb.open_file(self.outpath.format(file_number=self.out_file_number), "w")
                if self.verbosity:
                    print(">> Writing:", self.outpath.format(file_number=self.out_file_number), end="\n")

        # not enough space in current output file
        else:
            # finish to write current outputfile
            self.events = self.events_[:nidle].copy()

            self._select_data()

            df_writer(self.h5out, self.events, "Run", "events", 'ZLIB4')
            for key, table in self.tables.items():
                if key != "events":
                    exec(f"df_writer(self.h5out, self.{key}.reset_index(), *table, 'ZLIB4')")

            # write new file(s) (the loop avoids recursive calling of self._write_data)
            for fidx in np.arange(0, np.ceil((nevents-nidle)/self.nevents_per_file).astype(int)):

                init = int(nidle + self.nevents_per_file* fidx)
                last = int(nidle + self.nevents_per_file*(fidx+1)) # notice that len(self.events_) == nevents

                self.events = self.events_[init:last].copy()

                # open new file
                self.h5out.close()
                self.out_file_number += 1
                self.nevents_in_file  = 0
                self.h5out = tb.open_file(self.outpath.format(file_number=self.out_file_number), "w")
                if self.verbosity:
                    print(">> Writing:", self.outpath.format(file_number=self.out_file_number), end="\n")

                # write in new file
                self._select_data()
                df_writer(self.h5out, self.events, "Run", "events", 'ZLIB4')
                for key, table in self.tables.items():
                    if key != "events":
                        exec(f"df_writer(self.h5out, self.{key}.reset_index(), *table, 'ZLIB4')")

                self.nevents_in_file += len(self.events)
        return


    def _select_data(self):
        '''
        Selects the events from self.events and adds unique event-id:
        self.table_name = self.table_name.loc[self.events.index.evt_number)]
        (similar for MC data)
        '''
        for key, table in self.tables.items():
            if (key == "events"): continue

            elif (table[0] != "MC"):
                exec(f"self.{key} = self.{key}_.loc[self.events.evt_number]")

            elif (table[0] == "MC"): # notice that self.eventMap is created just before
                exec(f"self.{key} = self.{key}_.loc[self.{key}_.index.intersection(self.eventMap.nexus_evt)]")

        # add unique event-id to 'event' column
        indexes = ["evt_number", "G4Volume", "Isotope", "file"]
        self.events.set_index(indexes, inplace=True)

        n = len(self.events)
        self.events["event"] = self.saved_events + np.arange(0, n)
        self.saved_events += n

        for key, table in self.tables.items():
            if (key in ("dst", "tracking", "summary")):
                exec(f"self.{key}.set_index(indexes[1:], inplace=True, append=True)")
                exec(f"self.{key}['event'] = self.events.event")

        self.events.reset_index(inplace=True)
        return


    def _dispose_dfs(self):
        '''
        Deletes dataframe atributtes:
        del self.table_name
        del self.table_name_
        '''
        for key, table in self.tables.items():
            delattr(self, key)
            delattr(self, key + "_")
        return



def get_mixer_nevents(exposure : float, detector_db : str = "next100", isotopes : list = "all"):
    '''
    This function computes the number of events of each component (isotope, volume) pairs
    based on the activity-assumptions provided in the database.

    Parameters:
    ----------
    :exposure:    exposure time
    :detector_db: detector database
    :isotopes:    (default "all") list with the isotopes to simulate,
                  ignores signal-like events "0nubb" and "2nubb"

    Output:
    -------
    pandas.DataFrame object with three columns: (G4Volume, Isotope, nevents),
    where nevents is the number of events of each component for the given exposure
    '''

    # get activities and efficiencies from database
    act, eff = RadioactivityData(detector_db)

    # if a list of isotopes is provided, warn missing and select them
    if not (isotopes == "all"):
        # warn about missing isotopes in the database
        act_in = np.isin(isotopes, act.Isotope.unique())
        eff_in = np.isin(isotopes, eff.Isotope.unique())
        missing = ~(act_in | eff_in)

        if missing.any():
            isos = [iso for b, iso in zip(missing, isotopes) if b and (iso not in ["0nubb", "2nubb"])]
            if len(isos)>0:
                msg = f"Missing database isotopes: {isos}"
                warnings.warn(msg)

        # select requested isotopes
        act = act[act.Isotope.isin(isotopes)]
        eff = eff[eff.Isotope.isin(isotopes)]

    # warn about missing components
    act_uniq = act.value_counts(subset=["G4Volume", "Isotope"]).index
    eff_uniq = eff.value_counts(subset=["G4Volume", "Isotope"]).index

    act_in = act_uniq.isin(eff_uniq)
    if not act_in.all():
        msg = f"Components missing at Efficiency table: {str(act_uniq[act_in].to_list())}"
        warnings.warn(msg)

    eff_in = eff_uniq.isin(act_uniq)
    if not eff_in.all():
        msg = f"Components missing at Activity table: {str(eff_uniq[eff_in].to_list())}"
        warnings.warn(msg)

    # create nevents df and return it
    df = pd.merge(act, eff, on=["G4Volume", "Isotope"])
    df.loc[:, "nevts"] = (df.TotalActivity * mBq) * df.MCEfficiency * exposure
    df = df.drop(columns=["TotalActivity", "MCEfficiency"])
    return df


def get_reco_and_sim_nevents(inpath : str, components : list)->pd.DataFrame:
    '''
    This function computes the number of events of each component (isotope, volume) pairs
    that were simulated (nsim) and that have been reconstructed (nreco)

    Parameters:
    ----------
    :inpath:     path for the input files (see mixer.conf doc)
    :components: list of (G4Volume, Isotope) pairs

    Output:
    -------
    pandas.DataFrame object with four columns: (G4Volume, Isotope, nreco, nsim),
    where nreco is the number of reconstructed events and nsim the number of simulated events
    '''
    inpath  = os.path.expandvars(inpath)
    reco_df = pd.DataFrame(columns=["G4Volume", "Isotope", "nreco", "nsim"])

    for g4volume, isotope in components:
        filenames = glob.glob(inpath.format(g4volume=g4volume, isotope=isotope))
        nreco = 0
        nsim  = 0
        if len(filenames) == 0:
            raise Exception(f"Not files found for component: {g4volume}, {isotope}")
        for filename in filenames:
            events = load_dst(filename, "Run", "events")
            conf   = load_dst(filename,  "MC", "configuration").set_index("param_key")

            nreco += len(events)
            nsim  += int(conf.loc["saved_events", "param_value"])
        reco_df.loc[len(reco_df)] = (g4volume, isotope, nreco, nsim)

    return reco_df


def _check_enough_nevents(nevent_df : pd.DataFrame, eff_df : pd.DataFrame):
    '''
    Function to check that the number of events in the input
    data is enough to run the mixer
    '''
    indexes   = ["G4Volume", "Isotope"]
    nevent_df = nevent_df.set_index(indexes)
    eff_df    =    eff_df.set_index(indexes)
    sel       = (eff_df.nreco >= nevent_df.nevts)

    if sel.all(): return # enough events
    else:                # not enought events
        msg = "Not enough input data for: \n"
        for (g4volume, isotope) in nevent_df[~sel].index:
            nevts = nevent_df.loc[g4volume, isotope].nevts
            nreco =    eff_df.loc[g4volume, isotope].nreco
            msg += f"{g4volume}, {isotope}: required {nevts} got {nreco} \n"
        raise Exception(msg)
