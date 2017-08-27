
import tables
from ..evm.event_model     import MCParticle
from ..evm.event_model     import MCHit
from typing import Mapping

# use Mapping (duck type) rather than dict

def load_mchits(file_name: str, max_events:int =1e+9) -> Mapping[int, MCHit]:

    with tables.open_file(file_name,mode='r') as h5in:
        mctable = h5in.root.MC.MCTracks
        mcevents = read_mctracks (mctable, max_events)
        mchits_dict = compute_mchits_dict(mcevents)
    return mchits_dict


def read_mctracks (mc_table: tables.table.Table,
                   max_events:int =1e+9) ->Mapping[int, Mapping[int, MCParticle]]:

    all_events = {}
    current_event = {}
#    convert table to numpy.ndarray
    data       = mc_table[:]
    data_size  = len(data)

    event            =  data["event_indx"]
    particle         =  data["mctrk_indx"]
    particle_name    =  data["particle_name"]
    pdg_code         =  data["pdg_code"]
    initial_vertex   =  data["initial_vertex"]
    final_vertex     =  data["final_vertex"]
    momentum         =  data["momentum"]
    energy           =  data["energy"]
    nof_hits         =  data["nof_hits"]
    hit              =  data["hit_indx"]
    hit_position     =  data["hit_position"]
    hit_time         =  data["hit_time"]
    hit_energy       =  data["hit_energy"]

    for i in range(data_size):
        if event[i] >= max_events:
            break

        current_event = all_events.setdefault(event[i], {})

        current_particle = current_event.setdefault( particle[i],
                                                    MCParticle(particle_name[i],
                                                               pdg_code[i],
                                                               initial_vertex[i],
                                                               final_vertex[i],
                                                               momentum[i],
                                                               energy[i]))
        hit = MCHit(hit_position[i], hit_time[i], hit_energy[i])
        current_particle.hits.append(hit)

    return all_events


def compute_mchits_dict(mcevents:Mapping[int, Mapping[int, MCParticle]])->Mapping[int, MCHit]:
    """Returns all hits in the event"""
    mchits_dict = {}
    for event_no, particle_dict in mcevents.items():
        hits = []
        for particle_no in particle_dict.keys():
            particle = particle_dict[particle_no]
            if abs(particle.pdg) == 11:
                hits.extend(particle.hits)
        mchits_dict[event_no] = hits
    return mchits_dict
