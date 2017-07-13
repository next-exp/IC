from .. reco     import tbl_functions as tbl
from .. evm.nh5 import FEE
from .. sierpe   import fee as FE


def write_FEE_table(h5out):
    fee_table = _create_FEE_table(h5out)
    _store_FEE_table(fee_table)


def _create_FEE_table(h5out):
    # create a table to store Energy plane FEE
    FEE_group = h5out.create_group(h5out.root, "FEE")
    fee_table = h5out.create_table(FEE_group, "FEE", FEE,
                                   "EP-FEE parameters",
                                   tbl.filters("NOCOMPR"))
    return fee_table


def _store_FEE_table(fee_table):
    """Store the parameters of the EP FEE simulation."""
    row = fee_table.row
    row["OFFSET"]        = FE.OFFSET
    row["CEILING"]       = FE.CEILING
    row["PMT_GAIN"]      = FE.PMT_GAIN
    row["FEE_GAIN"]      = FE.FEE_GAIN
    row["R1"]            = FE.R1
    row["C1"]            = FE.C1
    row["C2"]            = FE.C2
    row["ZIN"]           = FE.Zin
    row["DAQ_GAIN"]      = FE.DAQ_GAIN
    row["NBITS"]         = FE.NBITS
    row["LSB"]           = FE.LSB
    row["NOISE_I"]       = FE.NOISE_I
    row["NOISE_DAQ"]     = FE.NOISE_DAQ
    row["t_sample"]      = FE.t_sample
    row["f_sample"]      = FE.f_sample
    row["f_mc"]          = FE.f_mc
    row["f_LPF1"]        = FE.f_LPF1
    row["f_LPF2"]        = FE.f_LPF2
    row.append()
    fee_table.flush()
