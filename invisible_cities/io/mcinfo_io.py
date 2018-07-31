
import tables as tb
import numpy  as np

from .. reco            import tbl_functions as tbl
from .. core            import system_of_units as units
from .. core.exceptions import SensorBinningNotFound

from .. evm.event_model import MCParticle
from .. evm.event_model import MCHit
from .. evm.event_model import Waveform

from .. evm.nh5 import MCGeneratorInfo
from .. evm.nh5 import MCExtentInfo
from .. evm.nh5 import MCHitInfo
from .. evm.nh5 import MCParticleInfo

from typing import Mapping

# use Mapping (duck type) rather than dict

units_dict = {'picosecond' : units.picosecond,  'ps' : units.picosecond,
              'nanosecond' : units.nanosecond,  'ns' : units.nanosecond,
              'microsecond': units.microsecond, 'mus': units.microsecond,
              'millisecond': units.millisecond, 'ms' : units.millisecond}


class mc_info_writer:
    """Write MC info to file."""
    def __init__(self, h5file, compression = 'ZLIB4'):

        self.h5file      = h5file
        self.compression = compression
        self._create_tables()
        self.reset()
        self.current_tables = None

        self.last_written_hit      = 0
        self.last_written_particle = 0
        self.first_extent_row      = True
        self.first_file            = True


    def reset(self):
        # last visited row
        self.last_row              = 0

    def _create_tables(self):
        """Create tables in MC group in file h5file."""
        if '/MC' in self.h5file:
            MC = self.h5file.root.MC
        else:
            MC = self.h5file.create_group(self.h5file.root, "MC")

        self.extent_table = self.h5file.create_table(MC, "extents",
                                                     description = MCExtentInfo,
                                                     title       = "extents",
                                                     filters     = tbl.filters(self.compression))

        # Mark column to index after populating table
        self.extent_table.set_attr('columns_to_index', ['evt_number'])

        self.hit_table = self.h5file.create_table(MC, "hits",
                                                  description = MCHitInfo,
                                                  title       = "hits",
                                                  filters     = tbl.filters(self.compression))

        self.particle_table = self.h5file.create_table(MC, "particles",
                                                       description = MCParticleInfo,
                                                       title       = "particles",
                                                       filters     = tbl.filters(self.compression))

        self.generator_table = self.h5file.create_table(MC, "generators",
                                                        description = MCGeneratorInfo,
                                                        title       = "generators",
                                                        filters     = tbl.filters(self.compression))



    def __call__(self, mctables: (tb.Table, tb.Table, tb.Table, tb.Table),
                 evt_number: int):
        if mctables is not self.current_tables:
            self.current_tables = mctables
            self.reset()

        extents = mctables[0]
        for iext in range(self.last_row, len(extents)):
            this_row = extents[iext]
            if this_row['evt_number'] < evt_number: continue
            if this_row['evt_number'] > evt_number: break
            if iext == 0:
                if self.first_file:
                    modified_hit          = this_row['last_hit']
                    modified_particle     = this_row['last_particle']
                    self.first_extent_row = False
                    self.first_file       = False
                else:
                    modified_hit          = this_row['last_hit'] + self.last_written_hit + 1
                    modified_particle     = this_row['last_particle'] + self.last_written_particle + 1
                    self.first_extent_row = False

            elif self.first_extent_row:
                previous_row          = extents[iext-1]
                modified_hit          = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit - 1
                modified_particle     = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle - 1
                self.first_extent_row = False
                self.first_file       = False
            else:
                previous_row      = extents[iext-1]
                modified_hit      = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit
                modified_particle = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle

            modified_row                  = self.extent_table.row
            modified_row['evt_number']    = evt_number
            modified_row['last_hit']      = modified_hit
            modified_row['last_particle'] = modified_particle
            modified_row.append()

            self.last_written_hit      = modified_hit
            self.last_written_particle = modified_particle
        self.extent_table.flush()

        hits, particles, generators = read_mcinfo_evt(mctables, evt_number, self.last_row)
        self.last_row += 1

        for h in hits:
            new_row = self.hit_table.row
            new_row['hit_position']  = h[0]
            new_row['hit_time']      = h[1]
            new_row['hit_energy']    = h[2]
            new_row['label']         = h[3]
            new_row['particle_indx'] = h[4]
            new_row['hit_indx']      = h[5]
            new_row.append()
        self.hit_table.flush()

        for p in particles:
            new_row = self.particle_table.row
            new_row['particle_indx']  = p[0]
            new_row['particle_name']  = p[1]
            new_row['primary']        = p[2]
            new_row['mother_indx']    = p[3]
            new_row['initial_vertex'] = p[4]
            new_row['final_vertex']   = p[5]
            new_row['initial_volume'] = p[6]
            new_row['final_volume']   = p[7]
            new_row['momentum']       = p[8]
            new_row['kin_energy']     = p[9]
            new_row['creator_proc']   = p[10]
            new_row.append()
        self.particle_table.flush()

        for g in generators:
            new_row = self.generator_table.row
            new_row['evt_number']    = g[0]
            new_row['atomic_number'] = g[1]
            new_row['mass_number']   = g[2]
            new_row['region']        = g[3]
            new_row.append()
        self.generator_table.flush()


def load_mchits(file_name: str,
                event_range=(0, int(1e9))) -> Mapping[int, MCHit]:

    with tb.open_file(file_name, mode='r') as h5in:
        mcevents    = read_mcinfo(h5in, event_range)
        mchits_dict = compute_mchits_dict(mcevents)

    return mchits_dict


def load_mcparticles(file_name: str,
                     event_range=(0, int(1e9))) -> Mapping[int, MCParticle]:

    with tb.open_file(file_name, mode='r') as h5in:
        return read_mcinfo(h5in, event_range)


def load_mcsensor_response(file_name: str,
                           event_range=(0, int(1e9))) -> Mapping[int, MCParticle]:

    with tb.open_file(file_name, mode='r') as h5in:
        return read_mcsns_response(h5in, event_range)


def read_mcinfo_evt (mctables: (tb.Table, tb.Table, tb.Table, tb.Table),
                     event_number: int, last_row=0) -> ([tb.Table], [tb.Table], [tb.Table]):
    h5extents    = mctables[0]
    h5hits       = mctables[1]
    h5particles  = mctables[2]
    h5generators = mctables[3] 

    particle_rows = []
    hit_rows      = []
    generator_rows = [] 

    event_range = (last_row, int(1e9))
    for iext in range(*event_range):
        this_row = h5extents[iext]
        if this_row['evt_number'] == event_number:
            # the indices of the first hit and particle are 0 unless the first event
            #  written is to be skipped: in this case they must be read from the extents
            ihit = ipart = 0
            if iext > 0:
                previous_row = h5extents[iext-1]

                ihit         = int(previous_row['last_hit']) + 1
                ipart        = int(previous_row['last_particle']) + 1

            ihit_end  = this_row['last_hit']
            ipart_end = this_row['last_particle']

            while ihit <= ihit_end:
                if len(h5hits) != 0:
                    hit_rows.append(h5hits[ihit])
                ihit += 1

            while ipart <= ipart_end:
                particle_rows.append(h5particles[ipart])
                ipart += 1

            # It is possible for the 'generators' dataset to have a different length compared to the 'extents' dataset. In particular, it may occur that the 'generators' dataset is empty. In this case, do not add any rows to 'generators'.
            if len(h5generators) == len(h5extents):
                generator_rows.append(h5generators[last_row])

            break

    return hit_rows, particle_rows, generator_rows


def read_mcinfo(h5f, event_range=(0, int(1e9))) ->Mapping[int, Mapping[int, MCParticle]]:
    mc_info = tbl.get_mc_info(h5f)

    h5extents = mc_info.extents

    events_in_file = len(h5extents)

    all_events = {}

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        current_event           = {}
        evt_number              = h5extents[iext]['evt_number']
        hit_rows, particle_rows, generator_rows = read_mcinfo_evt(mc_info, evt_number, iext)

        for h5particle in particle_rows:
            this_particle = h5particle['particle_indx']
            current_event[this_particle] = MCParticle(h5particle['particle_name'].decode('utf-8','ignore'),
                                                      h5particle['primary'],
                                                      h5particle['mother_indx'],
                                                      h5particle['initial_vertex'],
                                                      h5particle['final_vertex'],
                                                      h5particle['initial_volume'].decode('utf-8','ignore'),
                                                      h5particle['final_volume'].decode('utf-8','ignore'),
                                                      h5particle['momentum'],
                                                      h5particle['kin_energy'],
                                                      h5particle['creator_proc'].decode('utf-8','ignore'))

        for h5hit in hit_rows:
            ipart            = h5hit['particle_indx']
            current_particle = current_event[ipart]

            hit = MCHit(h5hit['hit_position'],
                        h5hit['hit_time'],
                        h5hit['hit_energy'],
                        h5hit['label'].decode('utf-8','ignore'))

            current_particle.hits.append(hit)

        evt_number             = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events


def compute_mchits_dict(mcevents:Mapping[int, Mapping[int, MCParticle]])->Mapping[int, MCHit]:
    """Returns all hits in the event"""
    mchits_dict = {}
    for event_no, particle_dict in mcevents.items():
        hits = []
        for particle_no in particle_dict.keys():
            particle = particle_dict[particle_no]
            hits.extend(particle.hits)
        mchits_dict[event_no] = hits
    return mchits_dict


def read_mcsns_response(h5f, event_range=(0, 1e9)) ->Mapping[int, Mapping[int, Waveform]]:

    h5config = h5f.root.MC.configuration

    bin_width_PMT  = None
    bin_width_SiPM = None
    for row in h5config:
        param_name = row['param_key'].decode('utf-8','ignore')
        if param_name.find('time_binning') >= 0:
            param_value = row['param_value'].decode('utf-8','ignore')
            numb, unit  = param_value.split()
            if param_name.find('Pmt') > 0:
                bin_width_PMT = float(numb) * units_dict[unit]
            elif param_name.find('SiPM') >= 0:
                bin_width_SiPM = float(numb) * units_dict[unit]


    if bin_width_PMT is None:
        raise SensorBinningNotFound
    if bin_width_SiPM is None:
        raise SensorBinningNotFound


    h5extents   = h5f.root.MC.extents

    try:
        h5f.root.MC.waveforms[0]
    except IndexError:
        print('Error: this file has no sensor response information.')

    h5waveforms = h5f.root.MC.waveforms

    last_line_of_event = 'last_sns_data'
    events_in_file     = len(h5extents)

    all_events = {}

    iwvf = 0
    if event_range[0] > 0:
        iwvf = h5extents[event_range[0]-1][last_line_of_event] + 1

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        current_event = {}

        iwvf_end          = h5extents[iext][last_line_of_event]
        current_sensor_id = h5waveforms[iwvf]['sensor_id']
        time_bins         = []
        charges           = []
        while iwvf <= iwvf_end:
            wvf_row   = h5waveforms[iwvf]
            sensor_id = wvf_row['sensor_id']

            if sensor_id == current_sensor_id:
                time_bins.append(wvf_row['time_bin'])
                charges.  append(wvf_row['charge'])
            else:
                bin_width = bin_width_PMT if current_sensor_id < 1000 else bin_width_SiPM
                times     = np.array(time_bins) * bin_width

                current_event[current_sensor_id] = Waveform(times, charges, bin_width)

                time_bins = []
                charges   = []
                time_bins.append(wvf_row['time_bin'])
                charges.append(wvf_row['charge'])

                current_sensor_id = sensor_id

            iwvf += 1

        bin_width = bin_width_PMT if current_sensor_id < 1000 else bin_width_SiPM
        times     = np.array(time_bins) * bin_width
        current_event[current_sensor_id] = Waveform(times, charges, bin_width)

        evt_number             = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events
