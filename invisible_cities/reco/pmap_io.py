from   invisible_cities.reco.nh5 import S12, S2Si
import invisible_cities.reco.tbl_functions as tbl

class PMapWriter:
    """Write PMAPS to file. """
    def __init__(self,
                 pmap_file       = None,
                 compression     = 'ZLIB4'):

        if not pmap_file:
            raise NoOutputFile('pmap file cannot be Null')

        self.pmap_file   =  pmap_file
        self.compression =  compression

        self._set_pmap_tables()

    def _set_pmap_tables(self):
        """Set the output file."""

        # create a group
        pmapsgroup = self.pmap_file.create_group(
            self.pmap_file.root, "PMAPS")

        # create tables to store pmaps
        self.s1t  = self.pmap_file.create_table(
            pmapsgroup, "S1", S12, "S1 Table",
            tbl.filters(self.compression))

        self.s2t  = self.pmap_file.create_table(
            pmapsgroup, "S2", S12, "S2 Table",
            tbl.filters(self.compression))

        self.s2sit = self.pmap_file.create_table(
            pmapsgroup, "S2Si", S2Si, "S2Si Table",
            tbl.filters(self.compression))

        self.s1t  .cols.event.create_index()
        self.s2t  .cols.event.create_index()
        self.s2sit.cols.event.create_index()

    def _store_s12(self, S12, st, event):
        row = st.row
        for i in S12:
            time = S12[i][0]
            ene  = S12[i][1]
            assert len(time) == len(ene)
            for j in range(len(time)):
                row["event"] = event
                row["peak"] = i
                row["time"] = time[j]
                row["ene"]  =  ene[j]
                row.append()

    def _store_s2si(self, S2Si, st, event):
        row = st.row
        for i in S2Si:
            sipml = S2Si[i]
            for sipm in sipml:
                nsipm = sipm[0]
                ene   = sipm[1]
                for j, E in enumerate(ene):
                    row["event"] = event
                    row["peak"]    = i
                    row["nsipm"]   = nsipm
                    row["nsample"] = j
                    row["ene"]     = E
                    row.append()

    def store_pmaps(self, event, S1, S2, S2Si):
        """Store PMAPS."""

        self._store_s12 (S1, self.s1t,  event)
        self._store_s12 (S2, self.s2t,  event)
        self._store_s2si(S2Si, self.s2sit, event)

    def flush(self):
        """Flush all tables"""
        self.s1t.flush()
        self.s2t.flush()
        self.s2sit.flush()
