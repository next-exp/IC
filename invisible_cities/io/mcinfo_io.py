
import tables

from .. reco     import tbl_functions as tbl

from ..evm.event_model     import MCParticle
from ..evm.event_model     import MCHit

from .. evm.nh5  import MCExtentInfo
from .. evm.nh5  import MCHitInfo
from .. evm.nh5  import MCParticleInfo

from typing import Mapping

# use Mapping (duck type) rather than dict

class mc_info_writer:
    """Write MC info to file."""
    def __init__(self, h5file, compression = 'ZLIB4'):

        self.h5file      = h5file
        self.compression = compression
        self._create_tables()
        # last visited row
        self.last_row = 0
        self.last_written_hit = 0
        self.last_written_particle = 0
        self.first_extent_row = True

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

    def __call__(self, mctables: (tables.Table, tables.Table, tables.Table),
                     evt_number: int):

        extents = mctables[0]
        for iext in range(self.last_row, len(extents)):
            if extents[iext]['evt_number'] < evt_number:
                continue
            if  extents[iext]['evt_number'] > evt_number:
                break
            if iext == 0:
                modified_hit = extents[iext]['last_hit']
                modified_particle = extents[iext]['last_particle']
            elif self.first_extent_row:
                previous_row = extents[iext-1]
                modified_hit = extents[iext]['last_hit']-previous_row['last_hit']+self.last_written_hit-1
                modified_particle = extents[iext]['last_particle']-previous_row['last_particle']+self.last_written_particle-1
                self.first_extent_row = False
            else:
                previous_row = extents[iext-1]
                modified_hit = extents[iext]['last_hit']-previous_row['last_hit']+self.last_written_hit
                modified_particle = extents[iext]['last_particle']-previous_row['last_particle']+self.last_written_particle

            modified_row = self.extent_table.row
            modified_row['evt_number'] = evt_number
            modified_row['last_hit'] = modified_hit
            modified_row['last_particle'] = modified_particle
            modified_row.append()

            self.last_written_hit = modified_hit
            self.last_written_particle = modified_particle
        self.extent_table.flush()

        hits, particles = read_mcinfo_evt_by_evt(mctables, evt_number, self.last_row)
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


def load_mchits(file_name: str,
                    event_range=(0,int(1e9))) -> Mapping[int, MCHit]:

    with tables.open_file(file_name,mode='r') as h5in:
        mcevents = read_mcinfo(h5in, event_range)
        mchits_dict = compute_mchits_dict(mcevents)

    return mchits_dict


def load_mcparticles(file_name: str,
                         event_range=(0,int(1e9))) -> Mapping[int, MCParticle]:

    with tables.open_file(file_name,mode='r') as h5in:
        return read_mcinfo(h5in, event_range)


def load_mcsensor_response(file_name: str,
                               event_range=(0,int(1e9))) -> Mapping[int, MCParticle]:

    with tables.open_file(file_name,mode='r') as h5in:
        return read_mcsns_response(h5in, event_range)


def read_mcinfo_evt_by_evt (mctables: (tables.Table, tables.Table, tables.Table),
                                event_number: int, last_row: int) -> ([tables.Table], [tables.Table]):
    h5extents   = mctables[0]
    h5hits      = mctables[1]
    h5particles = mctables[2]

    particle_rows = []
    hit_rows = []

    event_range = (last_row,int(1e9))
    for iext in range(*event_range):
        if h5extents[iext]['evt_number'] == event_number:
            # the indices of the first hit and particle are 0 unless the first event
            #  written is to be skipped: in this case they must be read from the extents
            ihit = 0; ipart = 0
            if iext > 0:
                ihit = h5extents[iext-1]['last_hit'] + 1
                ipart = h5extents[iext-1]['last_particle'] + 1

            ihit_end  = h5extents[iext]['last_hit']
            ipart_end = h5extents[iext]['last_particle']

            while ihit <= ihit_end:
                hit_rows.append(h5hits[ihit])
                ihit += 1

            while ipart <= ipart_end:
                particle_rows.append(h5particles[ipart])
                ipart += 1

            break

    return hit_rows, particle_rows


def read_mcinfo(h5f, event_range=(0,int(1e9))) ->Mapping[int, Mapping[int, MCParticle]]:
    h5extents   = h5f.root.MC.extents
    h5hits      = h5f.root.MC.hits
    h5particles = h5f.root.MC.particles

    mc_info = (h5extents, h5hits, h5particles)

    all_events = {}

    for iext in range(*event_range):
        if iext >= len(h5extents):
            break

        current_event = {}
        evt_number = h5extents[iext]['evt_number']
        hit_rows, particle_rows = read_mcinfo_evt_by_evt(mc_info, evt_number, iext)

        for h5particle in particle_rows:

            current_event[h5particle['particle_indx']] = MCParticle(h5particle['particle_name'].decode('utf-8','ignore'),
                                               h5particle['primary'],
                                               h5particle['mother_indx'],
                                               h5particle['initial_vertex'],
                                               h5particle['final_vertex'],
                                               h5particle['initial_volume'].decode('utf-8','ignore'),
                                               h5particle['final_volume'].decode('utf-8','ignore'),
                                               h5particle['momentum'],
                                               h5particle['kin_energy'],
                                               h5particle['creator_proc'])

        for h5hit in hit_rows:
            ipart = h5hit['particle_indx']
            current_particle = current_event[ipart]

            hit = MCHit(h5hit['hit_position'], h5hit['hit_time'],
                          h5hit['hit_energy'], h5hit['label'].decode('utf-8','ignore'))

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


def read_mcsns_response(h5f, event_range=(0,1e9)) ->Mapping[int, Mapping[int, MCParticle]]:

#    h5config = h5f.get('Run/configuration')
#    for row in h5config:
#        if row['param_key']

    h5extents = h5f.root.Run.extents
    h5waveforms = h5f.root.Run.waveforms

    last_line_of_event = 'last_sns_data'

    all_events = {}

    iwvf = 0
    if(event_range[0] > 0):
        iwvf = h5extents[event_range[0]-1][last_line_of_event]+1

    for iext in range(*event_range):
        if (iext >= len(h5extents)):
            break

        current_event = {}

        iwvf_end = h5extents[iext][last_line_of_event]
        current_sensor_id = h5waveforms[iwvf]['sensor_id']
        times = []
        charges = []
        while (iwvf <= iwvf_end):
            wvf_row = h5waveforms[iwvf]
            sensor_id = wvf_row['sensor_id']

            if (sensor_id == current_sensor_id):
                times.append(wvf_row['time_bin'])
                charges.append(wvf_row['charge'])
            else:
                current_event[current_sensor_id] = zip(times, charges)
                times = []
                charges = []
                times.append(wvf_row['time_bin'])
                charges.append(wvf_row['charge'])
                current_sensor_id = sensor_id

            iwvf += 1

        evt_number = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events



