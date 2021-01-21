"""Plink2 file readers.


In order to run this, you must build pgenlib from source. Pgenlib can be found here:
https://github.com/chrchang/plink-ng/tree/master/2.0/Python.
"""

import pathlib
import time

import numpy as np
import pandas as pd
import tqdm

try:
    import pgenlib
except ImportError as e:
    print(
        "There was an error importing pgenlib. In order to use this script, you must "
        "build from source. Clone plink-ng from https://github.com/chrchang/plink-ng "
        "and build pgenlib.\n"
        "\n"
        "Build instructions:\n"
        "cd plink-ng/2.0/Python\n"
        "python3 setup.py build_ext\n"
        "python3 setup.py install"
    )
    raise e
except Exception as e:
    raise e


def pgen_reader(pgen_file, output_file=None, ref_allele=0, impute=None) -> np.ndarray:
    """This function reads in a .pgen file and outputs it to a file as a numpy array.

    * Values of `-9` indicate a missing call.
    * This function is not memory-aware but uses dtypes of int8 for a minimally sized
    array.

    :param pgen_file:
    :param output_file: Optional file to output to
    :param ref_allele: The reference allele in the .pgen file. When `ref_allele = 1`,
        values of `0` and `2` are switched. If there is no `ref_allele1` in the .pgen
        file, this program will segfault.
    :param impute: To impute the magic numbers or not. Currently only supports "median".
    :return: A numpy array with each row being a separate subject, each column being a
        separate SNV. The column indices will align with their row in the .pvar file.
    """
    pgen_file = pathlib.Path(pgen_file)
    bytes_file = bytes(str(pgen_file.resolve()), "utf8")

    with pgenlib.PgenReader(bytes_file, raw_sample_ct=None, sample_subset=None) as pf:
        subject_count = pf.get_raw_sample_ct()
        variant_count = pf.get_variant_ct()
        blocks = []
        chunks = np.array_split(np.arange(0, variant_count, dtype=np.uint32), 500)

        for variant_idxs in tqdm.tqdm(chunks, desc="Reading pgen file chunks"):
            # buf = np.empty((len(variant_idxs), subject_count), np.int8)
            buf = np.empty((subject_count, len(variant_idxs)), np.int8)
            pf.read_list(variant_idxs, buf, allele_idx=np.int32(0), sample_maj=True)

            # Reverse 0 and 2. Segfaults if you attempt to use the allele_idx.
            if ref_allele == 0:
                major_allele_counts = buf == 2
                minor_allele_counts = buf == 0
                buf[major_allele_counts] = 0
                buf[minor_allele_counts] = 2
            if impute == "median":
                # TODO: efficiently impute the median
                buf[buf == -9] = 0
            elif not impute:
                pass
            else:
                raise NotImplementedError(f"{impute} imputation has not been implemented yet.")
            blocks.append(buf)

    print("Merging chunks into a coherent array")
    start_time = time.time()
    complete_array = np.hstack(blocks)
    print(f"Took {(time.time() - start_time) / 60:5.3f} minutes to hstack the chunks")
    if output_file:
        output_file = pathlib.Path(output_file)
        with open(output_file, "wb") as f:
            np.save(f, complete_array)
    return complete_array


def pvar_reader(pvar_file, info=False) -> pd.DataFrame:
    pvar_file = pathlib.Path(pvar_file)
    print("Reading in the Pvar file")
    use_cols = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL"]
    dtypes = {
        "CRHOM": str,
        "POS": str,
        "ID": str,
        "REF": str,
        "ALT": str,
        "QUAL": float,
    }
    if info:
        use_cols += ["INFO"]
        dtypes["INFO"] = str
    pvar_df = pd.read_csv(
        pvar_file,
        sep="\t",
        comment="#",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
        usecols=use_cols,
        dtype=dtypes,
        index_col=False,
        low_memory=False,
    )
    pvar_df["POS"] = pvar_df["POS"].astype(int)
    pvar_df["QUAL"] = pvar_df["QUAL"].astype(float)
    return pvar_df


def pvar_header_reader(pvar_file) -> str:
    pvar_file = pathlib.Path(pvar_file)
    with open(pvar_file, "r") as f:
        lines = []
        for line in f:
            if line.startswith("#"):
                lines.append(line)
            else:
                break
    return "".join(lines)


def psam_reader(psam_file) -> pd.DataFrame:
    psam_file = pathlib.Path(psam_file)
    df = pd.read_csv(
        psam_file,
        sep="\t",
    )
    rename_cols = dict()
    for col in df.columns:
        if col.startswith("#"):
            rename_cols[col] = col[1:]
    return df.rename(rename_cols, axis=1)


if __name__ == "__main__":
    pgen_file = "data/pre-plinked/ldpruned_data.pgen"
    out_file = "data/pre-plinked/full_plink2_matrix.npy"
    variant_call_array = pgen_reader(pgen_file, None)
    variant_info = pvar_reader("data/pre-plinked/ldpruned_data.pvar")
    sample_df = psam_reader("data/pre-plinked/ldpruned_data.psam")
