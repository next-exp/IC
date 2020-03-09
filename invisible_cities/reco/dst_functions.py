import numpy  as np

def dst_event_id_selection(data, event_ids):
    """Filter a DST by a list of event IDs.
    Parameters
    ----------
    data      : pandas DataFrame
        DST to be filtered.
    event_ids : list
        List of event ids that will be kept.
    Returns
    -------
    filterdata : pandas DataFrame
        Filtered DST
    """
    if 'event' in data:
        sel = np.isin(data.event.values, event_ids)
        return data[sel]
    else:
        print(r'DST does not have an "event" field. Data returned is unfiltered.')
        return data
