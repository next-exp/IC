"""Functions for DNN-based analysis
JR March 2017
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import tables as tb

from   invisible_cities.core.log_config import logger

def read_xyz_labels(files_in, nmax, evt_numbers):
    """Read the (x,y) labels from the specified input files
       evt_numbers: the list of events to be read
    """

    labels = []; levt_numbers = []
    tot_ev = 0
    for fin in files_in:

        fxy = tb.open_file(fin,'r')
        tracks = fxy.root.MC.MCTracks

        # get the arrays containing the true information
        event_indx = np.array(tracks[:]['event_indx'],dtype=np.int32)
        hit_energy = np.array(tracks[:]['hit_energy'],dtype=np.float32)
        hit_pos = np.array(tracks[:]['hit_position'],dtype=np.float32)

        # construct the label
        lbl = np.zeros(3)
        tot_energy = 0; i = 0
        rd_evt = 0

        # align the events properly and loop over table rows
        while(i < len(event_indx) and (event_indx[i] != evt_numbers[rd_evt])):
            i += 1
        while((nmax < 0 or tot_ev < nmax) and i < len(event_indx)):

            ev = event_indx[i]

            # ensure we stay synchronized with the specified event numbers list
            if(ev == evt_numbers[rd_evt]):

                # compute average hit_pos, weighted by hit_energy
                lbl = lbl + hit_energy[i] * hit_pos[i]
                tot_energy += hit_energy[i]

                # save and reset the label if we have reached the end of this event
                if(i >= len(event_indx)-1 or
                  (i < len(event_indx)-1 and event_indx[i+1] > ev)):
                    lbl /= tot_energy
                    tot_energy = 0
                    labels.append(lbl)
                    levt_numbers.append(ev)
                    lbl = np.zeros(3)
                    tot_ev += 1
                    rd_evt += 1

            i += 1

        fxy.close()

    # convert to numpy arrays and return
    labels = np.array(labels)
    levt_numbers = np.array(levt_numbers)
    return labels, levt_numbers


def read_pmaps(files_in, nmax, id_to_coords, max_slices, tbin):
    """Read all PMaps from the specified input files:

        files_in: the input files

        Returns 3 numpy arrays:
            maps: a list of 48x48 numpy arrays of SiPM responses, one for
                    each of N slices
            energies: contains N values of summed cathode (PMT) responses
                        one for each slice
            evt_numbers: contains a single value equal to the event number

        Therefore for each event we will have dimensions of:
            maps       -- [N, 48, 48]
            energies   -- [N]
            evt_numbers -- [1]
    """

    maps = []; energies = []; evt_numbers = []
    tot_ev = 0
    for fin in files_in:

        logger.info("Opening file: {0}".format(fin))
        fpmap = tb.open_file(fin,'r')
        s2maps = fpmap.root.PMAPS.S2
        s2simaps = fpmap.root.PMAPS.S2Si

        # loop over all events.
        rnum = 0; rnum_si = 0       # row number in table iteration
        ev = 0                      # processed event number
        evtnum = s2maps[0]['event'] # event number from file
        while(rnum < s2maps.nrows and (nmax < 0 or tot_ev < nmax)):

            logger.info("-- Attempting to process event {0} with rnum {1} of {2}...".format(evtnum, rnum, s2maps.nrows))

            # get the initial time for this event
            t0 = s2maps[rnum]['time']

            # create SiPM maps for this event.
            tmap = np.zeros((48, 48, max_slices), dtype=np.float32)
            eslices = np.zeros(max_slices)

            # loop over all peaks for this event
            while(rnum < s2maps.nrows and s2maps[rnum]['event'] == evtnum):

                # get the times and energies for this peak
                pknum = s2maps[rnum]['peak']
                times = []
                while(rnum < s2maps.nrows and
                      s2maps[rnum]['event'] == evtnum and
                      s2maps[rnum]['peak'] == pknum):

                    # get the time value
                    time = s2maps[rnum]['time']
                    times.append(time)

                    # calculate the bin number.
                    bb = int((time - t0) / tbin)

                    # add the energy to the energy array.
                    eslices[bb] += s2maps[rnum]['ene']

                    rnum += 1

                # get the amplitudes for each time value for each SiPM recorded
                while(rnum_si < s2simaps.nrows and
                      s2simaps[rnum_si]['event'] == evtnum and
                      s2simaps[rnum_si]['peak'] == pknum):

                    num_sipm = s2simaps[rnum_si]['nsipm']
                    for ti in times:

                        # calculate the bin number
                        bb = int((ti - t0) / tbin)

                        # consistency check
                        sipm_id = s2simaps[rnum_si]['nsipm']
                        if(sipm_id != num_sipm):
                            logger.error("ERROR: SiPM number inconsistency",
                                         "-- in event {0}, peak {1}".format(evtnum,pknum))

                        # add the amplitude to the map
                        sipm_amp = s2simaps[rnum_si]['ene']
                        [i, j] = (id_to_coords[sipm_id] + 235) / 10
                        tmap[np.int8(i),np.int8(j),bb] += sipm_amp

                        rnum_si += 1

            # add the event to the lists of maps and energies if
            #  it contains some information.
            if(np.max(tmap) != 0):
                maps.append(tmap)
                energies.append(eslices)
                evt_numbers.append(np.array([evtnum]))
                ev += 1; tot_ev += 1
                logger.info("** Added map for event {0}; file event number {1}".format(ev,evtnum))
            else:
                logger.warning("*** WARNING *** not adding SiPM map with 0 max charge.")

            # Set to the next event.
            if(rnum < s2maps.nrows):
                evtnum = s2maps[rnum]['event']

        fpmap.close()

    # return the set of maps, energies, and event numbers
    maps = np.array(maps);
    energies = np.array(energies);
    evt_numbers = np.array(evt_numbers);
    return maps, energies, evt_numbers;
