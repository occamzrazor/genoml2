# Copyright 2020 The GenoML Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Import the necessary packages
import subprocess
import sys
import pandas as pd
from pandas_plink import read_plink1_bin

# Define the munging class
import genoml.dependencies

class munging:
    def __init__(self, pheno_path, run_prefix="GenoML_data", impute_type="median", p_gwas=0.001, addit_path=None, gwas_path=None, geno_path=None):
        self.pheno_path = pheno_path
        self.run_prefix = run_prefix
        
        self.impute_type = impute_type
        self.p_gwas = p_gwas
        
        self.addit_path = addit_path
        self.gwas_path = gwas_path
        self.geno_path = geno_path
        
        self.pheno_df = pd.read_csv(pheno_path, engine='c')
        # Raise an error and exit if the phenotype file is not properly formatted
        try:
            if set(['ID','PHENO']).issubset(self.pheno_df.columns) == False:
                raise ValueError("""
                Error: It doesn't look as though your phenotype file is properly formatted. 
                Did you check that the columns are 'ID' and 'PHENO' and that controls=0 and cases=1?""")
        except ValueError as ve:
            print(ve)
            sys.exit()

        if (addit_path==None):
            print("No additional features as predictors? No problem, we'll stick to genotypes.")
            self.addit_df = None
        else:
            self.addit_df = pd.read_csv(addit_path, engine='c')

        if (gwas_path==None):
            print("So you don't want to filter on P values from external GWAS? No worries, we don't usually either (if the dataset is large enough).")
            self.gwas_df = None
        else:
            self.gwas_df = pd.read_csv(gwas_path, engine='c')
            
        if (geno_path==None):
            print("So no genotypes? Okay, we'll just use additional features provided for the predictions.")
        else:
            print("Pruning your data and exporting a reduced set of genotypes.")

    def plink_inputs(self):
        # Initializing some variables
        plink_exec = genoml.dependencies.check_plink()
        impute_type = self.impute_type
        addit_df = self.addit_df
        pheno_df = self.pheno_df

        outfile_h5 = self.run_prefix + ".dataForML.h5"
        pheno_df.to_hdf(outfile_h5, key='pheno', mode='w')

        if (self.geno_path != None):
        # Set the bashes
            bash1a = f"{plink_exec} --bfile " + self.geno_path + " --indep-pairwise 1000 50 0.1"
            bash1b = f"{plink_exec} --bfile " + self.geno_path + " --extract " + self.run_prefix + \
                ".p_threshold_variants.tab" + " --indep-pairwise 1000 50 0.1"
        # may want to consider outputting temp_genos to dir in run_prefix
            bash2 = f"{plink_exec} --bfile " + self.geno_path + \
                " --extract plink.prune.in --make-bed --out temp_genos"
            bash3 = "cut -f 2,5 temp_genos.bim > " + \
                self.run_prefix + ".variants_and_alleles.tab"
            bash4 = "rm plink.log"
            bash5 = "rm plink.prune.*"
            bash6 = "rm " + self.run_prefix + ".log"
        # Set the bash command groups
            cmds_a = [bash1a, bash2, bash3, bash4, bash5, bash6]
            cmds_b = [bash1b, bash2, bash3, bash4, bash5, bash6]

        if (self.gwas_path != None) & (self.geno_path != None):
            p_thresh = self.p_gwas
            gwas_df_reduced = self.gwas_df[['SNP', 'p']]
            snps_to_keep = gwas_df_reduced.loc[(
                gwas_df_reduced['p'] <= p_thresh)]
            outfile = self.run_prefix + ".p_threshold_variants.tab"
            snps_to_keep.to_csv(outfile, index=False, sep="\t")
            print(
                f"Your candidate variant list prior to pruning is right here: {outfile}.")

        if (self.gwas_path == None) & (self.geno_path != None):
            print(
                f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_a:
                subprocess.run(cmd, shell=True)

        if (self.gwas_path != None) & (self.geno_path != None):
            print(
                f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_b:
                subprocess.run(cmd, shell=True)

        if (self.geno_path != None):
            
            g = read_plink1_bin('temp_genos.bed')
            g_pruned = g.drop(['fid','father','mother','gender', 'trait', 'chrom', 'cm', 'pos','a1'])

            g_pruned = g_pruned.set_index({'sample':'iid','variant':'snp'})
            g_pruned.values = g_pruned.values.astype('int')

        # swap pandas-plink genotype coding to match .raw format...more about that below:

        # for example, assuming C in minor allele, alleles are coded in plink .raw labels homozygous for minor allele as 2 and homozygous for major allele as 0:
        #A A   ->    0   
        #A C   ->    1   
        #C C   ->    2
        #0 0   ->   NA

        # where as, read_plink1_bin flips these, with homozygous minor allele = 0 and homozygous major allele = 2
        #A A   ->    2   
        #A C   ->    1   
        #C C   ->    0
        #0 0   ->   NA

            two_idx = (g_pruned.values == 2)
            zero_idx = (g_pruned.values == 0)

            g_pruned.values[two_idx] = 0
            g_pruned.values[zero_idx] = 2

            g_pd = g_pruned.to_pandas()
            g_pd.reset_index(inplace=True)
            raw_df = g_pd.rename(columns={'sample': 'ID'})
        #    del raw_df.index.name
        #    del raw_df.columns.name
            
        # now, remove temp_genos
            bash_rm_temp = "rm temp_genos.*"
            print(bash_rm_temp)
            subprocess.run(bash_rm_temp, shell=True)

    # Checking the impute flag and execute
        # Currently only supports mean and median
        impute_list = ["mean", "median"]
        
        if (self.geno_path != None):

            if impute_type not in impute_list:
                return "The 2 types of imputation currently supported are 'mean' and 'median'"
            elif impute_type.lower() == "mean":
                raw_df = raw_df.fillna(raw_df.mean())
            elif impute_type.lower() == "median":
                raw_df = raw_df.fillna(raw_df.median())
            print("")
            print(
                f"You have just imputed your genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
            print("Now your genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(raw_df.describe())
            print("#"*70)
            print("")

    # Checking the imputation of non-genotype features

        if (self.addit_path != None):
            if impute_type not in impute_list:
                return "The 2 types of imputation currently supported are 'mean' and 'median'"
            elif impute_type.lower() == "mean":
                addit_df = addit_df.fillna(addit_df.mean())
            elif impute_type.lower() == "median":
                addit_df = addit_df.fillna(addit_df.median())
            print("")
            print(
                f"You have just imputed your non-genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
            print("Now your non-genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)
            print("")

            # Remove the ID column
            cols = list(addit_df.columns)
            cols.remove('ID')
            addit_df[cols]

            # Z-scale the features
            for col in cols:
                if (addit_df[col].min() == 0.0) and (addit_df[col].max() == 1.0):
                    print(col, "is likely a binary indicator or a proportion and will not be scaled, just + 1 all the values of this variable and rerun to flag this column to be scaled.")
                else:
                    addit_df[col] = (addit_df[col] - addit_df[col].mean())/addit_df[col].std(ddof=0)

            print("")
            print("You have just Z-scaled your non-genotype features, putting everything on a numeric scale similar to genotypes.")
            print("Now your non-genotype features might look a little closer to zero (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)

    # Saving out the proper HDF5 file
        if (self.geno_path != None):
            merged = raw_df.to_hdf(outfile_h5, key='geno')

        if (self.addit_path != None):
            merged = addit_df.to_hdf(outfile_h5, key='addit')

        if (self.geno_path != None) & (self.addit_path != None):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            geno = pd.read_hdf(outfile_h5, key="geno")
            addit = pd.read_hdf(outfile_h5, key="addit")
            temp = pd.merge(pheno, addit, on='ID', how='inner')
            merged = pd.merge(temp, geno, on='ID', how='inner')

        if (self.geno_path != None) & (self.addit_path == None):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            geno = pd.read_hdf(outfile_h5, key="geno")
            merged = pd.merge(pheno, geno, on='ID', how='inner')

        if (self.geno_path == None) & (self.addit_path != None):
            pheno = pd.read_hdf(outfile_h5, key="pheno")
            addit = pd.read_hdf(outfile_h5, key="addit")
            merged = pd.merge(pheno, addit, on='ID', how='inner')

        self.merged = merged
        merged.to_hdf(outfile_h5, key='dataForML')

        return merged
