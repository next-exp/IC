import tables as tb
import pandas as pd
from .. core.exceptions import UnknownDST

def load_dst(filename, group, node):
    with tb.open_file(filename) as h5in:
        try:
            table = getattr(getattr(h5in.root, group), node).read()
        except:
            print(f' error loading {filename}')
            raise UnknownDST

        return pd.DataFrame.from_records(table)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)

# def load_dsts(dst_list, group, node):
#     dsts=[]
#     for i, filename in enumerate(dst_list):
#         try:
#             dsts.append(dstio.load_dst(filename, group, node))
#         except:
#             print(f' error loading {filename}')
#
#     print(f'{i} files are being concatenated')
#     return pd.concat(dsts)
