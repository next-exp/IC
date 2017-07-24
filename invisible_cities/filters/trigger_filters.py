"""Trigger filter module
Compares each S2 in the PMTs selected for trigger with a set of cuts.
In order to pass the trigger filter, the S2 of a given PMT has to have
a charge (energy), height and width in the range defined by the cuts. The
trigger fires when a pre-specified number of PMTs pass the filter.
credits: see ic_authors_and_legal.rst in /doc
last revised: JJGC, July-2017

"""
def TriggerFilter(trigger_params):
    """Trigger Filter module"""
    def trigger_filter(peak_data : '{channel_no: s2}'):
        min_charge, max_charge = trigger_params.charge
        min_height, max_height = trigger_params.height
        min_width , max_width  = trigger_params. width
        n_channels_fired = 0
        # print(min_width,  max_width)
        # print(min_charge,  max_charge)
        # print(min_height,  max_height)

        for channel_no, s2 in peak_data.items():
            for peak_number, peak in s2.peaks.items():
                # print(peak.width)
                # print(peak.total_energy)
                # print(peak.height)
                if         min_width  < peak.width        <= max_width:
                    if     min_charge < peak.total_energy <= max_charge:
                        if min_height < peak.height <= max_height:
                            n_channels_fired += 1
                            # print(n_channels_fired)
                            if n_channels_fired == trigger_params.min_number_channels:
                                return True
        return False
    return trigger_filter
