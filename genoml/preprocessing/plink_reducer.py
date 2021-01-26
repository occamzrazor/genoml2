from genoml.preprocessing import plink2_reader
from graph_data import gencode
from typing import Optional, Dict, Tuple
import pgenlib
import numpy as np
import os
import pandas as pd


CHUNK_SIZE = int(1e5)
POSSIBLE_VEP_IMPACTS = ["HIGH", "MODERATE", "LOW", "MODIFIER"]


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

    def reduce_and_output(self, pfile_prefix: str):
        """Outputs all the pfiles according to plink format."""
        if not self.check_variants_computed():
            self.compute_variants()
        # Output psam
        outfile = pfile_prefix + 'reduced.psam'
        psam_file = plink2_reader.psam_reader(pfile_prefix + '.psam')
        with open(outfile, 'w') as f:
            psam_file.to_csv(f, sep="\t", index=False, header=None)
        # Output pvar
        outfile = pfile_prefix + 'reduced.pvar'
        self.reduce_pvar(outfile)
        # Output pgen
        self.reduce_pgen(self.variant_indices)

    def reduce_pvar(self, outfile: Optional[str] = None):
        """
        Reduces the pvar file by keeping the variants found by the compute_variants() function.
        """
        self.check_variants_computed(exception=True)
        reduced_pvar = self.pvar_df[self.variant_indices]
        print("Pvar file has been reduced!")
        if outfile:
            pvar_header = plink2_reader.pvar_header_reader(self.file_path + '.pvar')
            with open(outfile, "w") as f:
                f.write(pvar_header)
                reduced_pvar.to_csv(f, sep="\t", index=False, header=None)
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

    def check_variants_computed(self, raise_exception=False):
        check = self.variant_indices is None
        if check and raise_exception:
            raise Exception(
                "Please run `.compute_variants()` before running the "
                "`.reudce` commands"
            )
        return check


class RegionReducer(Plink2ReducerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chr_splits: Dict = self._get_splits()

    def compute_variants(self) -> np.ndarray:
        """Computes the variants to reduce the plink file to.

        :return: The variants to reduce the plink2 files to.
        """
        self.variant_indices = list()
        for chr, df in self.pvar_df.groupby(by="CHROM"):
            if chr not in self.chr_splits:
                self.variant_indices.append(self._handle_missing_chromosome_regions(df))
                continue
            lower_bound, upper_bound = self.chr_splits[chr]

            loci = df["POS"].to_numpy()
            lower_bound_check = lower_bound <= loci.reshape(-1, 1)
            upper_bound_check = loci.reshape(-1, 1) <= upper_bound
            variant_mask = self._get_keep_variant_mask(
                lower_bound_check, upper_bound_check
            )

            self.variant_indices.append(
                df[variant_mask].index.to_numpy(dtype=np.uint32)
            )
        self.variant_indices = np.concatenate(self.variant_indices)
        return self.variant_indices

    def _get_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """"""
        raise NotImplementedError

    @staticmethod
    def _handle_missing_chromosome_regions(df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _get_keep_variant_mask(lower_bound_check, upper_bound_check) -> np.ndarray:
        raise NotImplementedError


class InclusiveReducer(RegionReducer):
    @staticmethod
    def _handle_missing_chromosome_regions(df: pd.DataFrame) -> np.ndarray:
        """Ignore all variants when there are no inclusive regions."""
        return np.array([])

    @staticmethod
    def _get_keep_variant_mask(lower_bound_check, upper_bound_check) -> np.ndarray:
        """Only keep regions within and upper and lower bound region."""
        return (lower_bound_check & upper_bound_check).any(axis=1)


class ExclusiveReducer(RegionReducer):
    @staticmethod
    def _handle_missing_chromosome_regions(df: pd.DataFrame) -> np.ndarray:
        """Keep all variants if there are no exclusive regions."""
        return df.index.to_numpy(dtype=np.unint32)

    @staticmethod
    def _get_keep_variant_mask(lower_bound_check, upper_bound_check) -> np.ndarray:
        """Only include variants outside the exclusive bounds."""
        return ~(lower_bound_check & upper_bound_check).any(axis=1)


class GencodeReducer(InclusiveReducer):
    def __init__(
        self,
        *args,
        dataset_name: str = "gene_annotation",
        tss_distance: Optional[int] = None,
        **kwargs,
    ):
        dataset_name = dataset_name.lower()
        if dataset_name not in {"gene_annotation", "exon_annotation"}:
            raise ValueError(
                "The value of gencode must be one of: "
                '{"gene_annotation", "exon_annotation"}.\n'
                f"Not: {dataset_name}."
            )
        self.gencode_dataset_name = dataset_name
        self.__gencode_df = None
        self.tss_distance = tss_distance
        super().__init__(*args, **kwargs)

    @property
    def _gencode_df(self):
        if self.__gencode_df is not None:
            # We could return a copy but might take longer? Assume that users are going to
            # mutate this.
            return self.__gencode_df
        self.__gencode_df = get_gencode_data(self.gencode_dataset_name)
        if self.tss_distance and self.gencode_dataset_name == "gene_annotation":
            self.__gencode_df["lower_bound"] = (
                self.__gencode_df["TSS"] - self.tss_distance
            )
            self.__gencode_df["upper_bound"] = (
                self.__gencode_df["TSS"] + self.tss_distance
            )
        else:
            self.__gencode_df = self.__gencode_df.rename(
                {"start": "lower_bound", "end": "upper_bound"}, axis=1
            )

        return self.__gencode_df

    def _get_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        gencode_split = dict()
        for chr, df in self._gencode_df.groupby(by="CHROM"):
            upper_bound = df["upper_bound"].to_numpy(dtype=int)
            lower_bound = df["lower_bound"].to_numpy(dtype=int)
            gencode_split[chr] = (lower_bound, upper_bound)
        return gencode_split


class ENCODEBlackListReducer(ExclusiveReducer):
    def __init__(self, *args, encode_blacklist_file: str, **kwargs):
        self.encode_blacklist = pd.read_csv(
            encode_blacklist_file, sep="\t", names=["chr", "start", "stop"]
        )
        super().__init__(*args, **kwargs)

    def _get_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        splits = dict()
        for chr, blacklist in self.encode_blacklist.groupby(by="chr"):
            chr = chr[3:]  # Map to how plink has the CHROMs
            splits[chr] = (blacklist["start"].to_numpy(), blacklist["stop"].to_numpy())
        return splits


class VEPReducer(Plink2ReducerBase):
    def __init__(self, *args, lowest_impact: str = "HIGH", **kwargs):
        super().__init__(*args, **kwargs)
        if lowest_impact.upper() not in POSSIBLE_VEP_IMPACTS:
            raise ValueError(
                f"The value of lowest impact must be one of {POSSIBLE_VEP_IMPACTS}."
            )
        self.lowest_impact = lowest_impact

    def compute_variants(self) -> np.ndarray:
        self.variant_indices = list()
        for impact in POSSIBLE_VEP_IMPACTS:
            impact_variants = self.pvar_df[
                self.pvar_df["INFO"].str.lower().str.contains(impact.lower())
            ]
            self.variant_indices.append(impact_variants.index.to_numpy(dtype=np.uint32))
            if impact == self.lowest_impact:
                break

        self.variant_indices = np.concatenate(self.variant_indices)
        return self.variant_indices


def get_gencode_data(dataset_name: str) -> pd.DataFrame:
    """Gets the Gencode data from graph-data"""
    db = gencode.GencodeDatabase()
    db.install()
    db.load()
    gencode_data = db.dataset(dataset_name).data.copy()
    if dataset_name == "gene_annotation":
        gencode_data.sort_values(by="TSS", ignore_index=True)
    gencode_data["CHROM"] = gencode_data["chr"].str.strip("chr")
    return gencode_data


if __name__ == "__main__":
    # file_prefix = os.getcwd() + "/data/pre-plinked/ldpruned_data"
    file_prefix = "data/patient_data/ld_pruned/chr1/chr1_ldpruned"
    k = 5000

    gencode_tss_check = GencodeReducer(file_prefix=file_prefix, tss_distance=5e3)
    gencode_tss_variants = gencode_tss_check.compute_variants()
    gencode_tss_size = len(gencode_tss_variants)
    original_size = gencode_tss_check.pvar_df.shape[0]

    gencode_gene_check = GencodeReducer(file_prefix=file_prefix)
    gencode_gene_variants = gencode_gene_check.compute_variants()
    gencode_gene_size = len(gencode_gene_variants)

    gencode_exon_check = GencodeReducer(
        file_prefix=file_prefix, dataset_name="exon_annotation"
    )
    gencode_exon_variants = gencode_exon_check.compute_variants()
    gencode_exon_size = len(gencode_exon_variants)

    vep_check = VEPReducer(file_prefix=file_prefix, lowest_impact="moderate")
    vep_variants = vep_check.compute_variants()
    vep_size = len(vep_variants)

    encode_check = ENCODEBlackListReducer(
        file_prefix=file_prefix, encode_blacklist_file="data/hg38/ENCFF356LFX.bed.gz"
    )
    encode_variants = encode_check.compute_variants()
    encode_size = len(encode_variants)


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
