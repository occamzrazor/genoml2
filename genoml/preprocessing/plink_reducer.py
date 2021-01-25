from genoml.preprocessing import plink2_reader
from graph_data import gencode
from typing import Optional
import pgenlib
import numpy as np


class PlinkReducer(object):
    """
    Constructs and saves a reduced version of the Pgen file.
    The reduction is done, by finding variants in the Pvar file that satisfy some condition.
    The Pgen file is reduced to those variants.

    Usage:
    reducer = PlinkReducer(file_prefix, k)
    reduced_pvar = reducer.reduce_pvar(True)
    reducer.reduce_pgen(reduced_pvar)
    """
    def __init__(self, file_prefix: str, k: int):
        self.file_path = file_prefix
        self.k = k
        self.indices = None


    def get_genecode_data(self):
        """
        Returns genecode dataset in Graph Data.
        """
        db = gencode.GencodeDatabase()
        db.install()
        db.load()
        gencode_data = db.datasets[0].data.copy()
        gencode_data["CHROM"] = gencode_data["chr"].str.strip("chr")
        gencode_data["lower_bound"] = gencode_data["TSS"] - self.k
        gencode_data["upper_bound"] = gencode_data["TSS"] + self.k
        return gencode_data

    def pvar_file(self):
        """
        Returns Pvar file.
        """
        return plink2_reader.pvar_reader(self.file_path + ".pvar", info=False)

    def get_variants(self, df):
        """
        Finds pvar variants/indices to be kept, based on some condition.
        """
        pos = df['POS'].to_numpy()
        chrom = df['CHROM'].to_numpy()
        gencode_data = self.get_genecode_data()
        pvar_frame = []

        c = 0
        for _, row in gencode_data.iterrows():
            chromosome = str(row["CHROM"])
            lower = row["lower_bound"]
            upper = row["upper_bound"]
            pvar_sample = np.where((chrom == chromosome) & (pos >= lower) & (pos < upper))[0]
            pvar_frame.append(pvar_sample)
            c+=1
            if c>1:
                break
        self.indices = np.concatenate(pvar_frame).ravel()
        del gencode_data, pvar_frame
        return self.indices

    def reduce_pvar(self, outfile: Optional[bool] = False):
        """
        Reduces the pvar file by keeping the variants found by the get_variants() function.
        """
        df = self.pvar_file()
        indices = self.get_variants(df)
        df = df.to_numpy()
        reduced_pvar = df[indices]
        print('Pvar file has been reduced!')
        del df
        if outfile:
            outfile = self.file_path + 'reduced.pvar'
            with open(outfile, "wb") as f:
                np.save(f, reduced_pvar)
        return reduced_pvar

    def reduce_pgen(self, reduced_pvar: Optional[np.array]):
        """
        Reduces the pgen file by keeping the variants found by the get_variants() function.
        """
        print('Reducing Pgen File')
        pgen_file = self.file_path + '.pgen'
        bytes_file = bytes(pgen_file, "utf8")
        reduced_pgen = self.file_path + 'reduced.pgen'

        if reduced_pvar is not None:
            variant_ct = len(self.indices)
        else:
            df = self.pvar_file()
            variant_ct = len(self.get_variants(df))

        with pgenlib.PgenReader(bytes_file, raw_sample_ct=None, sample_subset=None) as infile:

            sample_ct = infile.get_raw_sample_ct()
            hardcall_phase_present = infile.hardcall_phase_present()

            with pgenlib.PgenWriter(bytes(reduced_pgen, 'utf8'), sample_ct, variant_ct, False,
                                    hardcall_phase_present=hardcall_phase_present) as outfile:
                if not hardcall_phase_present:
                    geno_buf = np.empty(sample_ct, np.int8)
                    for variant in range(variant_ct):
                        infile.read(variant, geno_buf, allele_idx=0)
                        outfile.append_biallelic(geno_buf)
                else:
                    allele_code_buf = np.empty(sample_ct * 2, np.int32)
                    phasepresent_buf = np.empty(sample_ct, np.bool_)
                    for variant in self.indices:
                        infile.read_alleles_and_phasepresent(variant, allele_code_buf, phasepresent_buf)
                        outfile.append_partially_phased(allele_code_buf, phasepresent_buf)
        print('Pgen file has been reduced!')
        return


if __name__ == "__main__":
    file_prefix = '/Users/mdmcastanos/genoml2/genoml/razor_training/data/pre-plinked/ldpruned_data'
    k = 5000
    reducer = PlinkReducer(file_prefix, k)
    reduced_pvar = reducer.reduce_pvar(True)
    reducer.reduce_pgen(reduced_pvar)


"""
Implement with dask dataframe:
    def reduce_pvar(self, outfile: Optional[str] = None):
        df = plink2_reader.pvar_reader(self.file_path + ".pvar", info=False)
        ddf = dd.from_pandas(df, npartitions=500).to_dask_array()
        gencode_data = self.get_genecode_data()
        pvar_frame = []
        for _, row in gencode_data.iterrows():
            chromosome = str(row["CHROM"])
            lower = row["lower_bound"]
            upper = row["upper_bound"]
            pvar_sample = ddf[(ddf["CHROM"] == chromosome) & ddf["POS"].between(lower, upper)] ## Do we always want this to be the metric?
            pvar_sample = pvar_sample.compute()
            pvar_frame.append(pvar_sample)

        if outfile:
            with open(outfile, "wb") as f:
                np.save(f, pvar_frame)

        return pd.concat(pvar_frame)
"""
