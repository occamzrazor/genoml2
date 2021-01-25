from genoml.preprocessing import plink2_reader
from graph_data import gencode
from typing import Optional, Dict, Tuple
import pgenlib
import numpy as np
import os
import pandas as pd


CHUNK_SIZE = int(1e5)


class Plink2ReducerBase(object):
    """
    Constructs and saves a reduced version of the Pgen file.
    The reduction is done, by finding variants in the Pvar file that satisfy some condition.
    The Pgen file is reduced to those variants.

    Usage:
    reducer = Plink2ReducerBase(file_prefix, k)
    reduced_pvar = reducer.reduce_pvar(True)
    reducer.reduce_pgen(reduced_pvar)
    """

    def __init__(self, file_prefix: str):
        self.file_path = file_prefix
        self.variant_indices = None
        self.__pvar_df = None

    @property
    def pvar_df(self):
        if self.__pvar_df is None:
            self.__pvar_df = plink2_reader.pvar_reader(
                self.file_path + ".pvar", info=True
            )
        # TODO Load this in with DASK.
        return self.__pvar_df

    def compute_variants(self) -> np.ndarray:
        """Computes the variants to reduce the plink file to.

        :return: The variants to reduce the plink2 files to.
        """
        raise NotImplementedError

    def reduce(self, output_prefix):
        # TODO (Maria)
        if not self.check_variants_computed():
            self.compute_variants()
        # Output psam

        # Output pvar

        # Output pgen

    def reduce_pvar(self, outfile: Optional[bool] = False):
        """
        Reduces the pvar file by keeping the variants found by the compute_variants() function.
        """
        df = self.pvar_df
        # indices = self.compute_variants(df)
        self.check_variants_computed(exception=True)
        # TODO(Maria): We cannot change the format of the pvar file, we want it to
        #  continue to be compatible with plink2. The new pvar file should look very
        #  similar to the old pvar files, with just the rows changed.
        df = df.to_numpy()
        reduced_pvar = df[indices]
        print("Pvar file has been reduced!")
        del df
        if outfile:
            outfile = self.file_path + "reduced.pvar"
            with open(outfile, "wb") as f:
                np.save(f, reduced_pvar)
        return reduced_pvar

    def reduce_pgen(self, keep_variants: np.array):
        """
        Reduces the pgen file by keeping the variants found by the compute_variants() function.
        """
        print("Reducing Pgen File")
        pgen_file = self.file_path + ".pgen"
        bytes_file = bytes(pgen_file, "utf8")
        reduced_pgen_bytes = bytes(self.file_path + "reduced.pgen", "utf8")

        keep_variants = keep_variants.astype(dtype=np.uint32)
        new_variant_ct = len(keep_variants)
        num_chunks = min(int(len(keep_variants) / CHUNK_SIZE), CHUNK_SIZE)
        chunks = np.split(keep_variants, num_chunks)

        with pgenlib.PgenReader(
            bytes_file, raw_sample_ct=None, sample_subset=None
        ) as infile:
            sample_ct = infile.get_raw_sample_ct()
            hardcall_phase_present = infile.hardcall_phase_present()

            with pgenlib.PgenWriter(
                reduced_pgen_bytes,
                sample_ct,
                new_variant_ct,
                False,
                hardcall_phase_present=hardcall_phase_present,
            ) as outfile:
                if not hardcall_phase_present:
                    for chunk in chunks:
                        geno_buf = np.empty((len(chunk), sample_ct), np.int8)
                        infile.read_list(chunk, geno_buf, allele_idx=0)
                        outfile.append_alleles_batch(geno_buf)
                else:
                    # Untested
                    for chunk in chunks:
                        allele_code_buf = np.empty(
                            (len(chunk), sample_ct * 2), np.uint32
                        )
                        phasepresent_buf = np.empty((len(chunk), sample_ct), np.bool_)
                        infile.read_alleles_and_phasepresent_list(
                            chunk, allele_code_buf, phasepresent_buf
                        )
                        outfile.append_partially_phased_batch(
                            allele_code_buf, phasepresent_buf
                        )
        print("Pgen file has been reduced!")

    def check_variants_computed(self, exception=False):
        check = self.variant_indices is None
        if check and exception:
            raise Exception(
                "Please run `.compute_variants()` before running the "
                "`.reudce_` commands"
            )
        return check


class GencodePromotorReducer(Plink2ReducerBase):
    def __init__(self, *args, k: int = 5000, **kwargs):
        super.__init__(*args, **kwargs)
        self.k = k
        self.__gencode_df = None

    @property
    def _gencode_df(self):
        if self.__gencode_df is None:
            self.__gencode_df = get_gencode_data()
            self.__gencode_df["lower_bound"] = self.__gencode_df["TSS"] - self.k
            self.__gencode_df["upper_bound"] = self.__gencode_df["TSS"] + self.k
        # We could return a copy but might take longer? Assume that users are going to
        # mutate this.
        return self.__gencode_df

    def get_genecode_by_chr(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        gencode_split = dict()
        for chr, df in self._gencode_df.groupby(by="CHROM"):
            upper_bound = df["upper_bound"].astype(int).to_numpy()
            lower_bound = df["lower_bound"].astype(int).to_numpy()
            gencode_split[chr] = (lower_bound, upper_bound)
        return gencode_split

    def compute_variants(self) -> np.ndarray:
        """Computes all variants +/-
        Note: Might be memory intensive.

        :return: All variant_indices within the ranges specified by Gencode.
        """
        gencode_split = self.get_genecode_by_chr()
        self.variant_indices = []
        for chr, df in self.pvar_df.groupby(by="CHROM"):
            lower_bound, upper_bound = gencode_split[chr]
            loci = df["POS"].to_numpy()
            lower_bound_check = lower_bound <= loci.reshape(-1, 1)
            upper_bound_check = loci.reshape(-1, 1) <= upper_bound
            bound_check = (lower_bound_check & upper_bound_check).any(axis=1)
            self.variant_indices.append(df[bound_check].index.to_numpy())
        self.variant_indices = np.concatenate(self.variant_indices)
        return self.variant_indices


def get_gencode_data() -> pd.DataFrame:
    """Gets the Gencode data from graph-data"""
    db = gencode.GencodeDatabase()
    db.install()
    db.load()
    gencode_data = db.datasets[0].data.copy()
    gencode_data.sort_values(by="TSS", ignore_index=True)
    gencode_data["CHROM"] = gencode_data["chr"].str.strip("chr")
    return gencode_data


if __name__ == "__main__":
    file_prefix = os.getcwd() + "/data/pre-plinked/ldpruned_data"
    k = 5000
    reducer = GencodePromotorReducer(file_prefix, k)
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
