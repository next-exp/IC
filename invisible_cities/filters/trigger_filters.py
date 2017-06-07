from collections import namedtuple

def TriggerFilter(trigger_params):
    def trigger_filter(data : '{channel_no: {peak_no: TriggerData}}'):
        min_charge, max_charge = trigger_params.charge_range
        min_height, max_height = trigger_params.height_range
        min_width , max_width  = trigger_params. width_range
        n_channels_fired = 0
        for channel_no, channel in data.items():
            for peak_no, td in channel.items():
                if         min_width  < td.width  <= max_width:
                    if     min_charge < td.charge <= max_charge:
                        if min_height < td.height <= max_height:
                            n_channels_fired += 1
                            if n_channels_fired == trigger_params.min_number_channels:
                                return True
        return False
    return trigger_filter
