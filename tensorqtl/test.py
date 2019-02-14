#import modin.pandas as pd
import pandas as pd
import numpy as np
import tensorqtl
# import tensorqtl.genotypeio as genotypeio
import genotypeio
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.clear_session()


# define paths to data
plink_prefix_path = 'GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered'
expression_bed = 'GEUVADIS.445_samples.expression.bed.gz'
covariates_file = 'GEUVADIS.445_samples.covariates.txt'
prefix = 'GEUVADIS.445_samples'

# load phenotypes and covariates
phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(expression_bed)
covariates_df = pd.read_csv(covariates_file, sep='\t', index_col=0).T

# PLINK reader for genotypes
pr = genotypeio.PlinkReader(plink_prefix_path, select_samples=phenotype_df.columns)


# map all cis-associations (results for each chromosome are written to file)

# all genes
# tensorqtl.map_cis_nominal(pr, phenotype_df, phenotype_pos_df, covariates_df, prefix)

# genes on chr18
tensorqtl.map_cis_nominal(pr, phenotype_df.loc[phenotype_pos_df['chr']=='chr18'],
                          phenotype_pos_df, covariates_df, prefix)

# load results
# pairs_df = pd.read_parquet('{}.variant_phenotype_pairs.chr18.parquet'.format(prefix))
# pairs_df.head()

# all genes
# cis_df = tensorqtl.map_cis(pr, phenotype_df, phenotype_pos_df, covariates_df)

# genes on chr18
cis_df = tensorqtl.map_cis(pr, phenotype_df.loc[phenotype_pos_df['chr']=='chr18'], phenotype_pos_df, covariates_df)


#cis_df.head()


genotype_df = genotypeio.load_genotypes(plink_prefix_path)


# run mapping
# to limit output size, only associations with p-value <= 1e-5 are returned
trans_df = tensorqtl.map_trans(genotype_df, phenotype_df, covariates_df, batch_size=50000,
                               return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05)


# remove cis-associations
trans_df = tensorqtl.filter_cis(trans_df, phenotype_pos_df.T.to_dict(), window=1000000)

trans_df.head()


