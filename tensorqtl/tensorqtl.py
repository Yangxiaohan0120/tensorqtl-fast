#!/usr/bin/env python3
from __future__ import print_function

import argparse
import glob
import gzip
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import scipy.optimize
import scipy.stats as stats
import tensorflow as tf
import tqdm
from scipy.special import loggamma

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio

has_rpy2 = False
e = subprocess.call('which R', shell=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
try:
    import rpy2
    import rfunc

    if e == 0:
        has_rpy2 = True
except:
    pass
if not has_rpy2:
    print(
        "Warning: 'rfunc' cannot be imported. R and the 'rpy2' Python package "
        "are needed.")

OMP_NUM_THREADS = 4
INTEROP = 2
INTRAOP = OMP_NUM_THREADS
KMP_BLOCKTIME = 0
KMP_SETTINGS = 1
KMP_AFFINITY = 'granularity=fine,verbose,compact,1,0'

os.environ["KMP_BLOCKTIME"] = str(KMP_BLOCKTIME)
os.environ["KMP_SETTINGS"] = str(KMP_SETTINGS)
os.environ["KMP_AFFINITY"] = KMP_AFFINITY
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


class SimpleLogger(object):
    def __init__(self, logfile=None, verbose=True):
        self.console = sys.stdout
        self.verbose = verbose
        if logfile is not None:
            self.log = open(logfile, 'w')
        else:
            self.log = None

    def write(self, message):
        if self.verbose:
            self.console.write(message + '\n')
        if self.log is not None:
            self.log.write(message + '\n')
            self.log.flush()


output_dtype_dict = {
    'num_var'     : np.int32,
    'beta_shape1' : np.float32,
    'beta_shape2' : np.float32,
    'true_df'     : np.float32,
    'pval_true_df': np.float64,
    'variant_id'  : str,
    'tss_distance': np.int32,
    'ma_samples'  : np.int32,
    'ma_count'    : np.int32,
    'maf'         : np.float32,
    'ref_factor'  : np.int32,
    'pval_nominal': np.float64,
    'slope'       : np.float32,
    'slope_se'    : np.float32,
    'pval_perm'   : np.float64,
    'pval_beta'   : np.float64, }


# ------------------------------------------------------------------------------
#  Core functions for mapping associations on GPU
# ------------------------------------------------------------------------------
class Residualizer(object):
    def __init__(self, C_t):
        # center and orthogonalize
        self.Q_t, _ = tf.qr(C_t - tf.reduce_mean(C_t, 0), full_matrices=False,
                            name='qr')

    def transform(self, M_t, center=True):
        """Residualize rows of M wrt columns of C"""
        if center:
            M0_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
        else:
            M0_t = M_t
        return M_t - tf.matmul(tf.matmul(M0_t, self.Q_t), self.Q_t,
                               transpose_b=True)  # keep original mean


def residualize(M_t, C_t):
    """Residualize M wrt columns of C"""

    # center and orthogonalize
    Q_t, _ = tf.qr(C_t - tf.reduce_mean(C_t, 0), full_matrices=False, name='qr')

    # residualize M relative to C
    M0_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
    return M_t - tf.matmul(tf.matmul(M0_t, Q_t), Q_t,
                           transpose_b=True)  # keep original mean


def center_normalize(M_t, axis=0):
    """Center and normalize M"""
    if axis == 0:
        N_t = M_t - tf.reduce_mean(M_t, 0)
        return tf.divide(N_t, tf.sqrt(tf.reduce_sum(tf.pow(N_t, 2), 0)))
    elif axis == 1:
        N_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
        return tf.divide(N_t, tf.sqrt(
            tf.reduce_sum(tf.pow(N_t, 2), axis=1, keepdims=True)))


def calculate_maf(genotype_t):
    """Calculate minor allele frequency"""
    af_t = tf.reduce_sum(genotype_t, 1) / (
            2 * tf.cast(tf.shape(genotype_t)[1], tf.float32))
    return tf.where(af_t > 0.5, 1 - af_t, af_t)


def _calculate_corr(genotype_t, phenotype_t, covariates_t, return_sd=False):
    """Calculate correlation between normalized residual genotypes and
    phenotypes"""
    # residualize
    genotype_res_t = residualize(genotype_t, covariates_t)  # variants x samples
    phenotype_res_t = residualize(phenotype_t,
                                  covariates_t)  # phenotypes x samples

    if return_sd:
        _, gstd = tf.nn.moments(genotype_res_t, axes=1)
        _, pstd = tf.nn.moments(phenotype_res_t, axes=1)

    # center and normalize
    genotype_res_t = center_normalize(genotype_res_t, axis=1)
    phenotype_res_t = center_normalize(phenotype_res_t, axis=1)

    # correlation
    if return_sd:
        return tf.squeeze(tf.matmul(genotype_res_t, phenotype_res_t,
                                    transpose_b=True)), tf.sqrt(pstd / gstd)
    else:
        return tf.squeeze(
            tf.matmul(genotype_res_t, phenotype_res_t, transpose_b=True))


def _calculate_max_r2(genotypes_t, phenotypes_t, permutations_t, covariates_t,
                      maf_threshold=0.05):
    maf_t = calculate_maf(genotypes_t)
    ix = tf.where(maf_t >= maf_threshold)

    g2 = tf.squeeze(tf.gather(genotypes_t, ix))
    r2_nom_t = tf.pow(_calculate_corr(g2, phenotypes_t, covariates_t), 2)
    r2_emp_t = tf.pow(_calculate_corr(g2, permutations_t, covariates_t), 2)

    return tf.squeeze(tf.reduce_max(r2_nom_t, axis=0)), tf.squeeze(
        tf.reduce_max(r2_emp_t, axis=0)), tf.gather(tf.squeeze(ix),
                                                    tf.argmax(r2_nom_t, axis=0))


def calculate_pval(r2_t, dof, maf_t=None, return_sparse=True, r2_threshold=0,
                   return_r2=False):
    """Calculate p-values from squared correlations"""
    dims = r2_t.get_shape()
    if return_sparse:
        ix = tf.where(r2_t >= r2_threshold, name='threshold_r2')
        r2_t = tf.gather_nd(r2_t, ix)

    r2_t = tf.cast(r2_t, tf.float64)

    tstat = tf.sqrt(tf.divide(tf.scalar_mul(dof, r2_t), 1 - r2_t), name='tstat')
    tdist = tf.contrib.distributions.StudentT(np.float64(dof),
                                              loc=np.float64(0.0),
                                              scale=np.float64(1.0))

    if return_sparse:
        pval_t = tf.SparseTensor(ix,
                                 tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat))),
                                 dims)
        if maf_t is not None:
            maf_t = tf.gather(maf_t, ix[:, 0])
    else:
        pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat)))

    if maf_t is not None:
        if return_r2:
            return pval_t, maf_t, r2_t
        else:
            return pval_t, maf_t
    else:
        return pval_t


def _interaction_assoc_row(genotype_t, phenotype_t, icovariates_t,
                           return_sd=False):
    """
    genotype_t must be a 1D tensor
    icovariates_t: [covariates_t, interaction_t]
    """
    gi_covariates_t = tf.concat(
        [icovariates_t, tf.reshape(genotype_t, [-1, 1])], axis=1)
    ix_t = tf.reshape(tf.multiply(genotype_t, icovariates_t[:, -1]),
                      [1, -1])  # must be 1 x N
    return _calculate_corr(ix_t, phenotype_t, gi_covariates_t,
                           return_sd=return_sd)


def calculate_association(genotype_t, phenotype_t, covariates_t,
                          interaction_t=None, return_sparse=True,
                          r2_threshold=None, return_r2=False):
    """Calculate genotype-phenotype associations"""
    maf_t = calculate_maf(genotype_t)

    if interaction_t is None:
        r2_t = tf.pow(_calculate_corr(genotype_t, phenotype_t, covariates_t), 2)
        dof = genotype_t.shape[1].value - 2 - covariates_t.shape[1].value
    else:
        icovariates_t = tf.concat([covariates_t, interaction_t], axis=1)
        r2_t = tf.pow(tf.map_fn(
            lambda x: _interaction_assoc_row(x, phenotype_t, icovariates_t),
            genotype_t, infer_shape=False), 2)
        dof = genotype_t.shape[1].value - 4 - covariates_t.shape[1].value

    return calculate_pval(r2_t, dof, maf_t, return_sparse=return_sparse,
                          r2_threshold=r2_threshold, return_r2=return_r2)


def get_sample_indexes(vcf_sample_ids, phenotype_df):
    """Get index of sample IDs in VCF"""
    return tf.constant([vcf_sample_ids.index(i) for i in phenotype_df.columns])


def _get_sample_indexes(vcf_sample_ids, phenotype_df):
    """Get index of sample IDs in VCF"""
    return [vcf_sample_ids.index(i) for i in phenotype_df.columns]


def initialize_data(phenotype_df, covariates_df, batch_size, interaction_s=None,
                    dtype=tf.float32):
    """Generate placeholders"""
    num_samples = phenotype_df.shape[1]
    genotype_t = tf.placeholder(dtype, shape=[batch_size, num_samples])
    phenotype_t = tf.constant(phenotype_df.values, dtype=dtype)
    phenotype_t = tf.reshape(phenotype_t, shape=[-1, num_samples])
    covariates_t = tf.constant(covariates_df.values, dtype=dtype)
    covariates_t = tf.reshape(covariates_t, shape=[-1, covariates_df.shape[1]])
    if interaction_s is None:
        return genotype_t, phenotype_t, covariates_t
    else:
        interaction_t = tf.constant(interaction_s.values, dtype=dtype)
        interaction_t = tf.reshape(interaction_t, [-1, 1])
        return genotype_t, phenotype_t, covariates_t, interaction_t


# ------------------------------------------------------------------------------
#  Functions for beta-approximating empirical p-values
# ------------------------------------------------------------------------------
def pval_from_corr(r2, dof):
    tstat2 = dof * r2 / (1 - r2)
    return 2 * stats.t.cdf(-np.abs(np.sqrt(tstat2)), dof)


def df_cost(r2, dof):
    """minimize abs(1-alpha) as a function of M_eff"""
    pval = pval_from_corr(r2, dof)
    mean = np.mean(pval)
    var = np.var(pval)
    return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0


def beta_log_likelihood(x, shape1, shape2):
    """negative log-likelihood of beta distribution"""
    logbeta = loggamma(shape1) + loggamma(shape2) - loggamma(shape1 + shape2)
    return (1.0 - shape1) * np.sum(np.log(x)) + (1.0 - shape2) * np.sum(
        np.log(1.0 - x)) + len(x) * logbeta


def fit_beta_parameters(r2_perm, dof, tol=1e-4, return_minp=False):
    """
      r2_perm:    array of max. r2 values from permutations
      dof:        degrees of freedom
    """
    try:
        true_dof = scipy.optimize.newton(lambda x: df_cost(r2_perm, x), dof,
                                         tol=tol, maxiter=50)
    except:
        print(
            'WARNING: scipy.optimize.newton failed to converge (running '
            'scipy.optimize.minimize)')
        res = scipy.optimize.minimize(lambda x: np.abs(df_cost(r2_perm, x)),
                                      dof, method='Nelder-Mead', tol=tol)
        true_dof = res.x[0]

    pval = pval_from_corr(r2_perm, true_dof)
    mean, var = np.mean(pval), np.var(pval)
    beta_shape1 = mean * (mean * (1 - mean) / var - 1)
    beta_shape2 = beta_shape1 * (1 / mean - 1)
    res = scipy.optimize.minimize(
        lambda s: beta_log_likelihood(pval, s[0], s[1]),
        [beta_shape1, beta_shape2], method='Nelder-Mead', tol=tol)
    beta_shape1, beta_shape2 = res.x
    if return_minp:
        return beta_shape1, beta_shape2, true_dof, pval
    else:
        return beta_shape1, beta_shape2, true_dof


def calculate_beta_approx_pval(r2_perm, r2_nominal, dof, tol=1e-4):
    """
      r2_nominal: nominal max. r2 (scalar or array)
      r2_perm:    array of max. r2 values from permutations
      dof:        degrees of freedom
    """
    beta_shape1, beta_shape2, true_dof = fit_beta_parameters(r2_perm, dof, tol)
    pval_true_dof = pval_from_corr(r2_nominal, true_dof)
    pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
    return pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof


# ------------------------------------------------------------------------------
#  Top-level functions for running cis-/trans-QTL mapping
# ------------------------------------------------------------------------------
def calculate_cis_permutations(genotypes_t, range_t, phenotype_t, covariates_t,
                               permutation_ix_t):
    """Calculate nominal and empirical correlations"""
    permutations_t = tf.gather(phenotype_t, permutation_ix_t)

    r_nominal_t, std_ratio_t = _calculate_corr(genotypes_t,
                                               tf.reshape(phenotype_t, [1, -1]),
                                               covariates_t, return_sd=True)

    corr_t = tf.pow(_calculate_corr(genotypes_t, permutations_t, covariates_t),
                    2)
    corr_t.set_shape([None, None])
    r2_perm_t = tf.cast(tf.reduce_max(
        tf.boolean_mask(corr_t, ~tf.reduce_any(tf.is_nan(corr_t), 1)), axis=0),
        tf.float64)

    ix = tf.argmax(tf.pow(r_nominal_t, 2))
    return r_nominal_t[ix], std_ratio_t[ix], range_t[ix], r2_perm_t, \
        genotypes_t[ix], tf.shape(r_nominal_t)[0]


def process_cis_permutations(r2_perm, r_nominal, std_ratio, g, num_var, dof,
                             n_samples, nperm=10000):
    """Calculate beta-approximated empirical p-value and annotate phenotype"""
    r2_nominal = r_nominal * r_nominal
    pval_perm = (np.sum(r2_perm >= r2_nominal) + 1) / (nperm + 1)
    pval_beta, beta_shape1, beta_shape2, true_dof, pval_true_dof = \
        calculate_beta_approx_pval(
            r2_perm, r2_nominal, dof)

    maf = np.sum(g) / (2 * n_samples)
    if maf <= 0.5:
        ref_factor = 1
        ma_samples = np.sum(g > 0.5)
        ma_count = np.sum(g[g > 0.5])
    else:
        maf = 1 - maf
        ref_factor = -1
        ma_samples = np.sum(g < 1.5)
        ma_count = np.sum(g[g < 1.5])

    slope = r_nominal * std_ratio
    tstat2 = dof * r2_nominal / (1 - r2_nominal)
    slope_se = np.abs(slope) / np.sqrt(tstat2)

    return pd.Series(OrderedDict(
        [('num_var', num_var), ('beta_shape1', beta_shape1),
            ('beta_shape2', beta_shape2), ('true_df', true_dof),
            ('pval_true_df', pval_true_dof), ('variant_id', np.NaN),
            ('tss_distance', np.NaN), ('ma_samples', ma_samples),
            ('ma_count', ma_count), ('maf', maf), ('ref_factor', ref_factor),
            ('pval_nominal', pval_from_corr(r2_nominal, dof)), ('slope', slope),
            ('slope_se', slope_se), ('pval_perm', pval_perm),
            ('pval_beta', pval_beta), ]))


def _process_group_permutations(buf):
    """
    Merge results for grouped phenotypes

    buf: [r_nom, s_r, var_ix, r2_perm, g, ng, nid]
    """
    # select phenotype with strongest nominal association
    max_ix = np.argmax(np.abs([b[0] for b in buf]))
    r_nom, s_r, var_ix = buf[max_ix][:3]
    g, ng, nid = buf[max_ix][4:]
    # select best phenotype correlation for each permutation
    r2_perm = np.max([b[3] for b in buf], 0)
    return r_nom, s_r, var_ix, r2_perm, g, ng, nid


def map_cis(plink_reader, phenotype_df, phenotype_pos_df, covariates_df,
            group_s=None, nperm=10000, logger=None):
    """Run cis-QTL mapping"""
    assert np.all(phenotype_df.columns == covariates_df.index)
    if logger is None:
        logger = SimpleLogger()

    logger.write('cis-QTL mapping: empirical p-values for phenotypes')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    if group_s is not None:
        logger.write('  * {} phenotype groups'.format(len(group_s.unique())))
        group_dict = group_s.to_dict()
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))

    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    permutation_ix_t = tf.convert_to_tensor(
        np.array([np.random.permutation(ix) for i in range(nperm)]))

    # placeholders
    covariates_t = tf.constant(covariates_df.values, dtype=tf.float32)
    genotype_t = tf.placeholder(dtype=tf.float32, shape=(None))
    phenotype_t = tf.placeholder(dtype=tf.float32, shape=(None))

    # iterate over chromosomes
    res_df = []
    start_time = time.time()

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = INTRAOP
    config.inter_op_parallelism_threads = INTEROP

    with tf.Session(config=config) as sess:
        for chrom in phenotype_pos_df.loc[phenotype_df.index, 'chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[
                phenotype_pos_df['chr'] == chrom], phenotype_pos_df)

            dataset = tf.data.Dataset.from_generator(igc.generate_data,
                                                     output_types=(
                                                         tf.float32, tf.float32,
                                                         tf.int32, tf.string))
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            next_phenotype, next_genotypes, next_range, next_id = \
                iterator.get_next()

            r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t, ng_t = \
                calculate_cis_permutations(
                    next_genotypes, next_range, next_phenotype, covariates_t,
                    permutation_ix_t)

            if group_s is None:
                for i in range(1, igc.n_phenotypes + 1):
                    r_nom, s_r, var_ix, r2_perm, g, ng, nid = sess.run(
                        [r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t,
                            ng_t, next_id])

                    # post-processing (on CPU)
                    res_s = process_cis_permutations(r2_perm, r_nom, s_r, g, ng,
                                                     dof, phenotype_df.shape[1],
                                                     nperm=nperm)
                    res_s.name = nid.decode()
                    res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                    res_s['tss_distance'] = igc.chr_variant_pos[
                                                res_s['variant_id']] - \
                                            igc.phenotype_tss[res_s.name]
                    res_df.append(res_s)
                    print(
                        '\r  * computing permutations for phenotype {}/{'
                        '}'.format(
                            i, igc.n_phenotypes), end='')
                print()
            else:
                n_groups = len(igc.phenotype_df.index.map(group_dict).unique())
                buf = []
                processed_groups = 0
                previous_group = None
                for i in range(0, igc.n_phenotypes):
                    group_id = group_dict[igc.phenotype_df.index[i]]
                    ires = sess.run(
                        [r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t,
                            ng_t, next_id])
                    if (group_id != previous_group and len(
                            buf) > 0):  # new group, process previous
                        # post-processing (on CPU)
                        r_nom, s_r, var_ix, r2_perm, g, ng, nid = \
                            _process_group_permutations(
                                buf)
                        res_s = process_cis_permutations(r2_perm, r_nom, s_r, g,
                                                         ng, dof,
                                                         phenotype_df.shape[1],
                                                         nperm=nperm)
                        res_s.name = nid.decode()
                        res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                        res_s['tss_distance'] = igc.chr_variant_pos[
                                                    res_s['variant_id']] - \
                                                igc.phenotype_tss[res_s.name]
                        res_s['group_id'] = group_id
                        res_s['group_size'] = len(buf)
                        res_df.append(res_s)
                        processed_groups += 1
                        print(
                            '\r  * computing permutations for phenotype group '
                            '{}/{}'.format(
                                processed_groups, n_groups), end='')
                        # reset
                        buf = [ires]
                    else:
                        buf.append(ires)
                    previous_group = group_id
                # process last group
                r_nom, s_r, var_ix, r2_perm, g, ng, nid = \
                    _process_group_permutations(
                        buf)
                res_s = process_cis_permutations(r2_perm, r_nom, s_r, g, ng,
                                                 dof, phenotype_df.shape[1],
                                                 nperm=nperm)
                res_s.name = nid.decode()
                res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                res_s['tss_distance'] = igc.chr_variant_pos[
                                            res_s['variant_id']] - \
                                        igc.phenotype_tss[res_s.name]
                res_s['group_id'] = group_id
                res_s['group_size'] = len(buf)
                res_df.append(res_s)
                processed_groups += 1
                print(
                    '\r  * computing permutations for phenotype group {}/{'
                    '}'.format(
                        processed_groups, n_groups), end='\n')

    res_df = pd.concat(res_df, axis=1).T
    res_df.index.name = 'phenotype_id'
    logger.write(
        '  Time elapsed: {:.2f} min'.format((time.time() - start_time) / 60))
    logger.write('done.')
    return res_df.astype(output_dtype_dict)


def map_cis_independent(plink_reader, summary_df, phenotype_df,
                        phenotype_pos_df, covariates_df, fdr=0.05,
                        fdr_col='qval', nperm=10000, logger=None):
    """
    Run independent cis-QTL mapping (forward-backward regression)

    summary_df: output from map_cis, annotated with q-values (calculate_qvalues)
    """
    assert np.all(phenotype_df.index == phenotype_pos_df.index)
    if logger is None:
        logger = SimpleLogger()

    signif_df = summary_df[summary_df[fdr_col] <= fdr].copy()
    signif_df = signif_df[
        ['num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df',
            'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf',
            'ref_factor', 'pval_nominal', 'slope', 'slope_se', 'pval_perm',
            'pval_beta', ]]
    signif_threshold = signif_df['pval_beta'].max()
    # subset significant phenotypes
    ix = signif_df.index[signif_df.index.isin(phenotype_df.index)]
    phenotype_df = phenotype_df.loc[ix]
    phenotype_pos_df = phenotype_pos_df.loc[ix]

    logger.write('cis-QTL mapping: conditionally independent variants')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} significant phenotypes'.format(signif_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))

    # print('Significance threshold: {}'.format(signif_threshold))

    # permutation indices
    n_samples = phenotype_df.shape[1]
    ix = np.arange(n_samples)
    permutation_ix_t = tf.convert_to_tensor(
        np.array([np.random.permutation(ix) for i in range(nperm)]))

    # placeholders
    genotypes_t = tf.placeholder(dtype=tf.float32, shape=[None, None])
    range_t = tf.placeholder(dtype=tf.int32, shape=[None])
    phenotype_t = tf.placeholder(dtype=tf.float32, shape=[None])
    covariates_t = tf.placeholder(dtype=tf.float32, shape=[None, None])

    # graph
    r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t, ng_t = \
        calculate_cis_permutations(
            genotypes_t, range_t, phenotype_t, covariates_t, permutation_ix_t)

    # iterate over chromosomes
    res_df = []
    start_time = time.time()
    with tf.Session() as sess:
        for chrom in phenotype_pos_df['chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[
                phenotype_pos_df['chr'] == chrom], phenotype_pos_df)
            ix_dict = {i: k for k, i in enumerate(plink_reader.bim.loc[
                                                      plink_reader.bim[
                                                          'chrom'] == chrom,
                                                      'snp'])}

            igc.chr_genotypes, igc.chr_variant_pos = \
                igc.plink_reader.get_region(
                    chrom, verbose=False)
            igc.loaded_chrom = chrom

            # iterate through significant phenotypes, fetch associated genotypes
            for j, (
                    phenotype, genotypes, genotype_range,
                    phenotype_id) in enumerate(
                igc.generate_data(), 1):

                # 1) forward pass
                forward_df = [signif_df.loc[
                    phenotype_id]]  # initialize results with top variant
                covariates = covariates_df.values.copy()  # initialize
                # covariates
                dosage_dict = {}
                while True:
                    # add variant to covariates
                    variant_id = forward_df[-1]['variant_id']
                    ig = igc.chr_genotypes[ix_dict[variant_id]]
                    dosage_dict[variant_id] = ig
                    covariates = np.hstack(
                        [covariates, ig.reshape(-1, 1)]).astype(np.float32)
                    dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                    # find next variant
                    r_nom, s_r, var_ix, r2_perm, g, ng = sess.run(
                        [r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t,
                            ng_t], feed_dict={genotypes_t: genotypes,
                            range_t                      : genotype_range,
                            phenotype_t                  : phenotype,
                            covariates_t                 : covariates})
                    res_s = process_cis_permutations(r2_perm, r_nom, s_r, g, ng,
                                                     dof, phenotype_df.shape[1],
                                                     nperm=nperm)
                    # add to list if significant
                    if res_s['pval_beta'] <= signif_threshold:
                        res_s.name = phenotype_id
                        res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                        res_s['tss_distance'] = igc.chr_variant_pos[
                                                    res_s['variant_id']] - \
                                                igc.phenotype_tss[res_s.name]
                        forward_df.append(res_s)
                    else:
                        break
                forward_df = pd.concat(forward_df, axis=1).T
                dosage_df = pd.DataFrame(dosage_dict)
                # print(forward_df)

                # 2) backward pass
                if forward_df.shape[0] > 1:
                    back_df = []
                    variant_set = set()
                    for k, i in enumerate(forward_df['variant_id'], 1):
                        covariates = np.hstack([covariates_df.values, dosage_df[
                            np.setdiff1d(forward_df['variant_id'],
                                         i)].values, ])
                        r_nom, s_r, var_ix, r2_perm, g, ng = sess.run(
                            [r_nominal_t, std_ratio_t, varpos_t, r2_perm_t, g_t,
                                ng_t], feed_dict={genotypes_t: genotypes,
                                range_t                      : genotype_range,
                                phenotype_t                  : phenotype,
                                covariates_t                 : covariates})
                        dof = phenotype_df.shape[1] - 2 - covariates.shape[1]
                        res_s = process_cis_permutations(r2_perm, r_nom, s_r, g,
                                                         ng, dof,
                                                         phenotype_df.shape[1],
                                                         nperm=nperm)
                        res_s['variant_id'] = igc.chr_variant_pos.index[var_ix]
                        if res_s['pval_beta'] <= signif_threshold and res_s[
                            'variant_id'] not in variant_set:
                            res_s.name = phenotype_id
                            res_s['tss_distance'] = igc.chr_variant_pos[
                                                        res_s['variant_id']] - \
                                                    igc.phenotype_tss[
                                                        res_s.name]
                            res_s['rank'] = k
                            back_df.append(res_s)
                            variant_set.add(res_s['variant_id'])
                    if len(back_df) > 0:
                        res_df.append(pd.concat(back_df,
                                                axis=1).T)  # print('back')
                        # print(pd.concat(back_df, axis=1).T)
                else:  # single variant
                    forward_df['rank'] = 1
                    res_df.append(forward_df)
                print(
                    '\r  * computing independent QTL for phenotype {}/{'
                    '}'.format(
                        j, igc.n_phenotypes), end='')
            print()
    res_df = pd.concat(res_df, axis=0)
    res_df.index.name = 'phenotype_id'
    logger.write(
        '  Time elapsed: {:.2f} min'.format((time.time() - start_time) / 60))
    logger.write('done.')
    return res_df.reset_index().astype(output_dtype_dict)


def calculate_qvalues(res_df, fdr=0.05, qvalue_lambda=None):
    """Annotate permutation results with q-values, p-value threshold"""

    print('Computing q-values')
    print('  * Number of phenotypes tested: {}'.format(res_df.shape[0]))
    print(
        '  * Correlation between Beta-approximated and empirical p-values: : '
        '{:.4f}'.format(
            stats.pearsonr(res_df['pval_perm'], res_df['pval_beta'])[0]))

    # calculate q-values
    if qvalue_lambda is None:
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'])
    else:
        print('  * Calculating q-values with lambda = {:.3f}'.format(
            qvalue_lambda))
        qval, pi0 = rfunc.qvalue(res_df['pval_beta'], qvalue_lambda)
    res_df['qval'] = qval
    print('  * Proportion of significant phenotypes (1-pi0): {:.2f}'.format(
        1 - pi0))
    print('  * QTL phenotypes @ FDR {:.2f}: {}'.format(fdr, np.sum(
        res_df['qval'] <= fdr)))

    # determine global min(p) significance threshold and calculate nominal
    # p-value threshold for each gene
    ub = res_df.loc[res_df['qval'] > fdr, 'pval_beta'].sort_values()[0]
    lb = res_df.loc[res_df['qval'] <= fdr, 'pval_beta'].sort_values()[-1]
    pthreshold = (lb + ub) / 2
    print('  * min p-value threshold @ FDR {}: {:.6g}'.format(fdr, pthreshold))
    res_df['pval_nominal_threshold'] = stats.beta.ppf(pthreshold,
                                                      res_df['beta_shape1'],
                                                      res_df['beta_shape2'])


def annotate_genes(gene_df, annotation_gtf, lookup_df=None):
    """
    Add gene and variant annotations (e.g., gene_name, rs_id, etc.) to
    gene-level output

    gene_df:        output from map_cis()
    annotation_gtf: gene annotation in GTF format
    lookup_df:      DataFrame with variant annotations, indexed by 'variant_id'
    """
    gene_dict = {}
    print('[' + datetime.now().strftime(
        "%b %d %H:%M:%S") + '] Adding gene and variant annotations', flush=True)
    print('  * parsing GTF', flush=True)
    with open(annotation_gtf) as gtf:
        for row in gtf:
            row = row.strip().split('\t')
            if row[0][0] == '#' or row[2] != 'gene':
                continue
            # get gene_id and gene_name from attributes
            attr = dict(
                [i.split() for i in row[8].replace('"', '').split(';') if
                    i != ''])
            # gene_name, gene_chr, gene_start, gene_end, strand
            gene_dict[attr['gene_id']] = [attr['gene_name'], row[0], row[3],
                row[4], row[6]]

    print('  * annotating genes', flush=True)
    if 'group_id' in gene_df:
        gene_info = pd.DataFrame(
            data=[gene_dict[i] for i in gene_df['group_id']],
            columns=['gene_name', 'gene_chr', 'gene_start', 'gene_end',
                'strand'], index=gene_df.index)
    else:
        gene_info = pd.DataFrame(data=[gene_dict[i] for i in gene_df.index],
                                 columns=['gene_name', 'gene_chr', 'gene_start',
                                     'gene_end', 'strand'], index=gene_df.index)
    gene_df = pd.concat([gene_info, gene_df], axis=1)
    assert np.all(gene_df.index == gene_info.index)

    col_order = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand',
        'num_var', 'beta_shape1', 'beta_shape2', 'true_df', 'pval_true_df',
        'variant_id', 'tss_distance']
    if lookup_df is not None:
        print('  * adding variant annotations from lookup table', flush=True)
        gene_df = gene_df.join(lookup_df,
                               on='variant_id')  # add variant information
        col_order += list(lookup_df.columns)
    col_order += ['ma_samples', 'ma_count', 'maf', 'ref_factor', 'pval_nominal',
        'slope', 'slope_se', 'pval_perm', 'pval_beta']
    if 'group_id' in gene_df:
        col_order += ['group_id', 'group_size']
    col_order += ['qval', 'pval_nominal_threshold']
    gene_df = gene_df[col_order]
    print('done.', flush=True)
    return gene_df


def get_significant_pairs(res_df, nominal_prefix, fdr=0.05):
    """Significant variant-phenotype pairs based on nominal p-value threshold
    for each phenotype"""
    print('[' + datetime.now().strftime(
        "%b %d %H:%M:%S") + '] tensorQTL: filtering significant '
                            'variant-phenotype pairs',
          flush=True)
    assert 'qval' in res_df

    # significant phenotypes (apply FDR threshold)
    df = res_df.loc[
        res_df['qval'] <= fdr, ['pval_nominal_threshold', 'pval_nominal',
            'pval_beta']].copy()
    df.rename(columns={'pval_nominal': 'min_pval_nominal'}, inplace=True)
    signif_phenotype_ids = set(df.index)
    threshold_dict = df['pval_nominal_threshold'].to_dict()

    nominal_files = {os.path.basename(i).split('.')[-2]: i for i in
        glob.glob(nominal_prefix + '*.parquet')}
    chroms = sorted(nominal_files.keys())
    signif_df = []
    for k, c in enumerate(chroms, 1):
        print(
            '  * parsing significant variant-phenotype pairs for chr. {}/{'
            '}'.format(
                k, len(chroms)), end='\r', flush=True)
        nominal_df = pd.read_parquet(nominal_files[c])
        nominal_df = nominal_df[
            nominal_df['phenotype_id'].isin(signif_phenotype_ids)]

        m = nominal_df['pval_nominal'] < nominal_df['phenotype_id'].apply(
            lambda x: threshold_dict[x])
        signif_df.append(nominal_df[m])
    print()
    signif_df = pd.concat(signif_df, axis=0)
    signif_df = signif_df.merge(df, left_on='phenotype_id', right_index=True)
    print('[' + datetime.now().strftime("%b %d %H:%M:%S") + '] done',
          flush=True)
    return signif_df  # signif_df.to_parquet(nominal_prefix.rsplit('.',
    # 1)[0]+'.cis_qtl_significant_pairs.parquet')


def calculate_cis_nominal(genotypes_t, phenotype_t, covariates_t, dof):
    """
    Calculate nominal associations

    genotypes_t: genotypes x samples
    phenotype_t: single phenotype
    covariates_t: covariates matrix, samples x covariates
    """
    p = tf.reshape(phenotype_t, [1, -1])

    r_nominal_t, std_ratio_t = _calculate_corr(genotypes_t, p, covariates_t,
                                               return_sd=True)
    r2_nominal_t = tf.pow(r_nominal_t, 2)
    pval_t = calculate_pval(r2_nominal_t, dof, maf_t=None, return_sparse=False)
    slope_t = tf.multiply(r_nominal_t, std_ratio_t)
    slope_se_t = tf.divide(tf.abs(slope_t), tf.sqrt(
        tf.divide(tf.scalar_mul(dof, r2_nominal_t), 1 - r2_nominal_t)))

    # calculate MAF
    n = covariates_t.shape[0].value
    n2 = 2 * n
    af_t = tf.reduce_sum(genotypes_t, 1) / n2
    ix_t = af_t <= 0.5
    maf_t = tf.where(ix_t, af_t, 1 - af_t)
    # calculate MA samples and counts
    m = tf.cast(genotypes_t > 0.5, tf.float32)
    a = tf.reduce_sum(m, 1)
    b = tf.reduce_sum(tf.cast(genotypes_t < 1.5, tf.float32), 1)
    ma_samples_t = tf.where(ix_t, a, b)
    m = tf.multiply(m, genotypes_t)
    a = tf.reduce_sum(m, 1)
    ma_count_t = tf.where(ix_t, a, n2 - a)

    return pval_t, slope_t, slope_se_t, maf_t, ma_samples_t, ma_count_t


def map_cis_nominal(plink_reader, phenotype_df, phenotype_pos_df, covariates_df,
                    prefix, output_dir='.', logger=None):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    """
    if logger is None:
        logger = SimpleLogger()

    logger.write(
        'cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))

    # placeholders
    covariates_t = tf.constant(covariates_df.values, dtype=tf.float32)
    genotype_t = tf.placeholder(dtype=tf.float32, shape=(None))
    phenotype_t = tf.placeholder(dtype=tf.float32, shape=(None))
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]

    with tf.Session() as sess:
        # iterate over chromosomes
        start_time = time.time()
        for chrom in phenotype_pos_df.loc[phenotype_df.index, 'chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[
                phenotype_pos_df['chr'] == chrom], phenotype_pos_df)

            dataset = tf.data.Dataset.from_generator(igc.generate_data,
                                                     output_types=(
                                                         tf.float32, tf.float32,
                                                         tf.int32, tf.string))
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            next_phenotype, next_genotypes, _, next_id = iterator.get_next()
            x = calculate_cis_nominal(next_genotypes, next_phenotype,
                                      covariates_t, dof)

            chr_res_df = []
            for i in range(1, igc.n_phenotypes + 1):
                (pval_nominal, slope, slope_se, maf, ma_samples,
                ma_count), phenotype_id = sess.run([x, next_id])
                phenotype_id = phenotype_id.decode()
                r = igc.cis_ranges[phenotype_id]
                variant_ids = plink_reader.variant_pos[chrom].index[
                r[0]:r[1] + 1]
                nv = len(variant_ids)
                tss_distance = np.int32(
                    plink_reader.variant_pos[chrom].values[r[0]:r[1] + 1] -
                    igc.phenotype_tss[phenotype_id])

                chr_res_df.append(pd.DataFrame(OrderedDict(
                    [('phenotype_id', [phenotype_id] * nv),
                        ('variant_id', variant_ids),
                        ('tss_distance', tss_distance), ('maf', maf),
                        ('ma_samples', np.int32(ma_samples)),
                        ('ma_count', np.int32(ma_count)),
                        ('pval_nominal', pval_nominal), ('slope', slope),
                        ('slope_se', slope_se), ])))
                print(
                    '\r    computing associations for phenotype {}/{}'.format(i,
                                                                              igc.n_phenotypes),
                    end='')
            print()
            logger.write('    time elapsed: {:.2f} min'.format(
                (time.time() - start_time) / 60))
            print('  * writing output')
            pd.concat(chr_res_df, copy=False).to_parquet(
                os.path.join(output_dir,
                             '{}.cis_qtl_pairs.{}.parquet'.format(prefix,
                                                                  chrom)))
    logger.write('done.')


def calculate_nominal_interaction(genotypes_t, phenotype_t, interaction_t, dof,
                                  residualizer, interaction_mask_t=None,
                                  maf_threshold_interaction=0.05,
                                  return_sparse=False, tstat_threshold=None):
    """"""
    phenotype_t = tf.reshape(phenotype_t, [1, -1])
    # filter monomorphic sites (to avoid colinearity in X)
    mask_t = ~(tf.reduce_all(tf.equal(genotypes_t, 0), axis=1) | tf.reduce_all(
        tf.equal(genotypes_t, 1), axis=1) | tf.reduce_all(
        tf.equal(genotypes_t, 2), axis=1))

    if interaction_mask_t is not None:
        upper_t = tf.boolean_mask(genotypes_t, interaction_mask_t, axis=1)
        lower_t = tf.boolean_mask(genotypes_t, ~interaction_mask_t, axis=1)
        mask_t = (mask_t & (
                calculate_maf(upper_t) >= maf_threshold_interaction) & (
                          calculate_maf(
                              lower_t) >= maf_threshold_interaction))

    mask_t.set_shape([None])  # required for tf.boolean_mask
    genotypes_t = tf.boolean_mask(genotypes_t, mask_t)

    s = tf.shape(genotypes_t)
    ng = s[0]
    ns = tf.cast(s[1], tf.float32)

    g0_t = genotypes_t - tf.reduce_mean(genotypes_t, axis=1, keepdims=True)
    gi_t = tf.multiply(genotypes_t, interaction_t)
    gi0_t = gi_t - tf.reduce_mean(gi_t, axis=1, keepdims=True)

    i0_t = interaction_t - tf.reduce_mean(interaction_t)
    p_t = tf.reshape(phenotype_t, [1, -1])
    p0_t = p_t - tf.reduce_mean(p_t, axis=1, keepdims=True)

    # residualize rows
    g0_t = residualizer.transform(g0_t, center=False)
    gi0_t = residualizer.transform(gi0_t, center=False)
    p0_t = residualizer.transform(p0_t, center=False)
    i0_t = residualizer.transform(i0_t, center=False)
    i0_t = tf.tile(i0_t, [ng, 1])

    # regression
    X_t = tf.stack([g0_t, i0_t, gi0_t], axis=2)
    Xinv = tf.linalg.inv(tf.matmul(X_t, X_t, transpose_a=True))
    b_t = tf.matmul(tf.matmul(Xinv, X_t, transpose_b=True),
                    tf.tile(tf.reshape(p0_t, [1, 1, -1]), [ng, 1, 1]),
                    transpose_b=True)

    # calculate b, b_se
    r_t = tf.squeeze(tf.matmul(X_t, b_t)) - p0_t
    rss_t = tf.reduce_sum(tf.multiply(r_t, r_t), axis=1)
    Cx = Xinv * tf.reshape(rss_t, [-1, 1, 1]) / dof
    b_se_t = tf.sqrt(tf.matrix_diag_part(Cx))
    b_t = tf.squeeze(b_t)
    tstat_t = tf.divide(tf.cast(b_t, tf.float64), tf.cast(b_se_t, tf.float64))
    # weird tf bug? without cast/copy, divide appears to modify b_se_t??
    # calculate pval
    tdist = tf.contrib.distributions.StudentT(np.float64(dof),
                                              loc=np.float64(0.0),
                                              scale=np.float64(1.0))
    pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat_t)))

    # calculate MAF
    n2 = 2 * ns
    af_t = tf.reduce_sum(genotypes_t, 1) / n2
    ix_t = af_t <= 0.5
    maf_t = tf.where(ix_t, af_t, 1 - af_t)
    # calculate MA samples and counts
    m = tf.cast(genotypes_t > 0.5, tf.float32)
    a = tf.reduce_sum(m, 1)
    b = tf.reduce_sum(tf.cast(genotypes_t < 1.5, tf.float32), 1)
    ma_samples_t = tf.where(ix_t, a, b)
    m = tf.multiply(m, genotypes_t)
    a = tf.reduce_sum(m, 1)
    ma_count_t = tf.where(ix_t, a, n2 - a)

    return pval_t, b_t, b_se_t, maf_t, ma_samples_t, ma_count_t, mask_t


def map_cis_interaction_nominal(plink_reader, phenotype_df, phenotype_pos_df,
                                covariates_df, interaction_s, prefix,
                                maf_threshold_interaction=0.05, best_only=False,
                                output_dir='.', logger=None):
    """
    cis-QTL mapping: nominal associations for all variant-phenotype pairs

    Association results for each chromosome are written to parquet files
    in the format <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    """
    if logger is None:
        logger = SimpleLogger()

    logger.write(
        'cis-QTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write('  * {} samples'.format(phenotype_df.shape[1]))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(plink_reader.bed.shape[0]))
    logger.write('  * including interaction term')

    covariates_t = tf.constant(covariates_df.values, dtype=tf.float32)
    dof = phenotype_df.shape[1] - 4 - covariates_df.shape[1]
    interaction_t = tf.constant(interaction_s.values.reshape(1, -1),
                                dtype=tf.float32)  # 1 x n
    interaction_mask_t = tf.constant(interaction_s >= interaction_s.median())
    residualizer = Residualizer(covariates_t)

    with tf.Session() as sess:
        # iterate over chromosomes
        start_time = time.time()
        best_assoc = []
        for chrom in phenotype_pos_df.loc[phenotype_df.index, 'chr'].unique():
            logger.write('  Mapping chromosome {}'.format(chrom))
            igc = genotypeio.InputGeneratorCis(plink_reader, phenotype_df.loc[
                phenotype_pos_df['chr'] == chrom], phenotype_pos_df)

            dataset = tf.data.Dataset.from_generator(igc.generate_data,
                                                     output_types=(
                                                         tf.float32, tf.float32,
                                                         tf.int32, tf.string))
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()
            next_phenotype, next_genotypes, _, next_id = iterator.get_next()

            x = calculate_nominal_interaction(next_genotypes, next_phenotype,
                                              interaction_t, dof, residualizer,
                                              interaction_mask_t=interaction_mask_t,
                                              maf_threshold_interaction=0.05)

            chr_res_df = []
            for i in range(1, igc.n_phenotypes + 1):
                (pval, b, b_se, maf, ma_samples, ma_count,
                maf_mask), phenotype_id = sess.run([x, next_id])

                phenotype_id = phenotype_id.decode()
                r = igc.cis_ranges[phenotype_id]
                variant_ids = plink_reader.variant_pos[chrom].index[
                r[0]:r[1] + 1]
                tss_distance = np.int32(
                    plink_reader.variant_pos[chrom].values[r[0]:r[1] + 1] -
                    igc.phenotype_tss[phenotype_id])
                if maf_mask is not None:
                    variant_ids = variant_ids[maf_mask]
                    tss_distance = tss_distance[maf_mask]
                nv = len(variant_ids)
                df = pd.DataFrame(OrderedDict(
                    [('phenotype_id', [phenotype_id] * nv),
                        ('variant_id', variant_ids),
                        ('tss_distance', tss_distance), ('maf', maf),
                        ('ma_samples', np.int32(ma_samples)),
                        ('ma_count', np.int32(ma_count)),
                        ('pval_g', pval[:, 0]), ('b_g', b[:, 0]),
                        ('b_g_se', b_se[:, 0]), ('pval_i', pval[:, 1]),
                        ('b_i', b[:, 1]), ('b_i_se', b_se[:, 1]),
                        ('pval_gi', pval[:, 2]), ('b_gi', b[:, 2]),
                        ('b_gi_se', b_se[:, 2]), ]))
                if best_only:
                    best_assoc.append(df.loc[df['pval_gi'].idxmin()])
                else:
                    chr_res_df.append(df)
                print(
                    '\r    computing associations for phenotype {}/{}'.format(i,
                                                                              igc.n_phenotypes),
                    end='')
            print()
            logger.write('    time elapsed: {:.2f} min'.format(
                (time.time() - start_time) / 60))
            if not best_only:
                print('  * writing output')
                pd.concat(chr_res_df, copy=False).to_parquet(
                    os.path.join(output_dir,
                                 '{}.cis_qtl_pairs.{}.parquet'.format(prefix,
                                                                      chrom)))
        if best_only:
            pd.concat(best_assoc, axis=1).T.set_index('phenotype_id').to_csv(
                os.path.join(output_dir,
                             '{}.cis_qtl_top_assoc.txt.gz'.format(prefix)),
                sep='\t', float_format='%.6g', compression='gzip')
    logger.write('done.')


"""
==============================================================================================================================================================
==============================================================================================================================================================
Target function
==============================================================================================================================================================
==============================================================================================================================================================
"""


# @ray.remote
# def worker_task(ps, phenotype_df, covariates_df, interaction_s,
#                 batch_size, return_sparse,pval_threshold,return_r2):
#     g_iter = ray.get(ps.fetch_g_iter.remote())
#
#     n_samples = phenotype_df.shape[1]

# -------------------------------------------------------------------------

def _initialize_data(phenotype_df, covariates_df, batch_size,
                     interaction_s=None,
                     dtype=tf.float32):
    """Generate placeholders"""
    num_samples = phenotype_df.shape[1]
    genotype_t = tf.placeholder(dtype, shape=[batch_size, num_samples])
    phenotype_t = tf.constant(phenotype_df.values, dtype=dtype)
    phenotype_t = tf.reshape(phenotype_t, shape=[-1, num_samples])
    covariates_t = tf.constant(covariates_df.values, dtype=dtype)
    covariates_t = tf.reshape(covariates_t,
                              shape=[-1, covariates_df.shape[1]])
    if interaction_s is None:
        return genotype_t, phenotype_t, covariates_t
    else:
        interaction_t = tf.constant(interaction_s.values, dtype=dtype)
        interaction_t = tf.reshape(interaction_t, [-1, 1])
        return genotype_t, phenotype_t, covariates_t, interaction_t


# -------------------------------------------------------------------------

def initialize_data(phenotype_df, covariates_df, batch_size,
                    interaction_s=None,
                    dtype=tf.float32):
    """Generate placeholders"""
    num_samples = phenotype_df.shape[1]
    genotype_t = tf.placeholder(dtype, shape=[batch_size, num_samples])
    phenotype_t = tf.constant(phenotype_df.values, dtype=dtype)
    phenotype_t = tf.reshape(phenotype_t, shape=[-1, num_samples])
    covariates_t = tf.constant(covariates_df.values, dtype=dtype)
    covariates_t = tf.reshape(covariates_t,
                              shape=[-1, covariates_df.shape[1]])
    if interaction_s is None:
        return genotype_t, phenotype_t, covariates_t
    else:
        interaction_t = tf.constant(interaction_s.values, dtype=dtype)
        interaction_t = tf.reshape(interaction_t, [-1, 1])
        return genotype_t, phenotype_t, covariates_t, interaction_t


# -------------------------------------------------------------------------

def _calculate_maf(genotype_t):
    """Calculate minor allele frequency"""
    af_t = tf.reduce_sum(genotype_t, 1) / (
            2 * tf.cast(tf.shape(genotype_t)[1], tf.float32))
    return tf.where(af_t > 0.5, 1 - af_t, af_t)


# -------------------------------------------------------------------------

def _calculate_pval(r2_t, dof, maf_t=None, return_sparse=True,
                    r2_threshold=0,
                    return_r2=False):
    """Calculate p-values from squared correlations"""
    dims = r2_t.get_shape()
    if return_sparse:
        ix = tf.where(r2_t >= r2_threshold, name='threshold_r2')
        r2_t = tf.gather_nd(r2_t, ix)

    r2_t = tf.cast(r2_t, tf.float64)

    tstat = tf.sqrt(tf.divide(tf.scalar_mul(dof, r2_t), 1 - r2_t),
                    name='tstat')
    tdist = tf.contrib.distributions.StudentT(np.float64(dof),
                                              loc=np.float64(0.0),
                                              scale=np.float64(1.0))

    if return_sparse:
        pval_t = tf.SparseTensor(ix,
                                 tf.scalar_mul(2,
                                               tdist.cdf(-tf.abs(tstat))),
                                 dims)
        if maf_t is not None:
            maf_t = tf.gather(maf_t, ix[:, 0])
    else:
        pval_t = tf.scalar_mul(2, tdist.cdf(-tf.abs(tstat)))

    if maf_t is not None:
        if return_r2:
            return pval_t, maf_t, r2_t
        else:
            return pval_t, maf_t
    else:
        return pval_t


# -------------------------------------------------------------------------

def _residualize(M_t, C_t):
    """Residualize M wrt columns of C"""

    # center and orthogonalize
    Q_t, _ = tf.qr(C_t - tf.reduce_mean(C_t, 0), full_matrices=False,
                   name='qr')

    # residualize M relative to C
    M0_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
    return M_t - tf.matmul(tf.matmul(M0_t, Q_t), Q_t,
                           transpose_b=True)  # keep original mean


# -------------------------------------------------------------------------

def _center_normalize(M_t, axis=0):
    """Center and normalize M"""
    if axis == 0:
        N_t = M_t - tf.reduce_mean(M_t, 0)
        return tf.divide(N_t, tf.sqrt(tf.reduce_sum(tf.pow(N_t, 2), 0)))
    elif axis == 1:
        N_t = M_t - tf.reduce_mean(M_t, axis=1, keepdims=True)
        return tf.divide(N_t, tf.sqrt(
            tf.reduce_sum(tf.pow(N_t, 2), axis=1, keepdims=True)))


# -------------------------------------------------------------------------

def _calculate_corr(genotype_t, phenotype_t, covariates_t,
                    return_sd=False):
    """Calculate correlation between normalized residual genotypes and
    phenotypes"""
    # residualize
    genotype_res_t = _residualize(genotype_t,
                                  covariates_t)  # variants x samples
    phenotype_res_t = _residualize(phenotype_t,
                                   covariates_t)  # phenotypes x samples

    if return_sd:
        _, gstd = tf.nn.moments(genotype_res_t, axes=1)
        _, pstd = tf.nn.moments(phenotype_res_t, axes=1)

    # center and normalize
    genotype_res_t = _center_normalize(genotype_res_t, axis=1)
    phenotype_res_t = _center_normalize(phenotype_res_t, axis=1)

    # correlation
    if return_sd:
        return tf.squeeze(tf.matmul(genotype_res_t, phenotype_res_t,
                                    transpose_b=True)), tf.sqrt(pstd /
                                                                gstd)
    else:
        return tf.squeeze(
            tf.matmul(genotype_res_t, phenotype_res_t, transpose_b=True))


def _calculate_association(genotype_t, phenotype_t, covariates_t,
                           interaction_t=None, return_sparse=True,
                           r2_threshold=None, return_r2=False):
    """Calculate genotype-phenotype associations"""
    maf_t = _calculate_maf(genotype_t)

    if interaction_t is None:
        r2_t = tf.pow(
            _calculate_corr(genotype_t, phenotype_t, covariates_t), 2)
        dof = genotype_t.shape[1].value - 2 - covariates_t.shape[1].value
    else:
        icovariates_t = tf.concat([covariates_t, interaction_t], axis=1)
        r2_t = tf.pow(tf.map_fn(
            lambda x: _interaction_assoc_row(x, phenotype_t,
                                             icovariates_t),
            genotype_t, infer_shape=False), 2)
        dof = genotype_t.shape[1].value - 4 - covariates_t.shape[1].value

    return _calculate_pval(r2_t, dof, maf_t, return_sparse=return_sparse,
                           r2_threshold=r2_threshold, return_r2=return_r2)


@ray.remote
def worker_task(ps, phenotype_df, covariates_df, interaction_s,
                batch_size, return_sparse, pval_threshold, return_r2):
    g_iter = ray.get(ps.fetch_g_iter.remote())

    # Precision change test
    g_iter = np.float32(g_iter)

    if interaction_s is None:
        genotypes, phenotypes, covariates = _initialize_data(phenotype_df,
                                                             covariates_df,
                                                             batch_size=batch_size,
                                                             dtype=tf.float32)
    else:
        genotypes, phenotypes, covariates, interaction = initialize_data(
            phenotype_df, covariates_df, batch_size=batch_size,
            interaction_s=interaction_s)

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = INTRAOP
    config.inter_op_parallelism_threads = INTEROP

    sess = tf.Session(config=config)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess.run(init_op)

    n_samples = phenotype_df.shape[1]

    if return_sparse:
        dof = n_samples - 2 - covariates_df.shape[1]
        tstat_threshold = stats.t.ppf(pval_threshold / 2, dof)
        r2_threshold = tstat_threshold ** 2 / (dof + tstat_threshold ** 2)
    else:
        r2_threshold = None

    if interaction_s is None:
        x = _calculate_association(genotypes, phenotypes, covariates,
                                   return_sparse=return_sparse,
                                   r2_threshold=r2_threshold,
                                   return_r2=return_r2)
    else:
        x = _calculate_association(genotypes, phenotypes, covariates,
                                   interaction_t=interaction,
                                   return_sparse=return_sparse,
                                   r2_threshold=r2_threshold)

    output = sess.run(x, feed_dict={genotypes: g_iter})

    del g_iter, genotypes, phenotypes, covariates
    return output


@ray.remote
class ParameterServer(object):

    def __init__(self, genotype_df, phenotype_df, covariates_df,
                 interaction_s=None,
                 return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
                 return_r2=False, batch_size=20000, logger=None, verbose=True):
        """Run trans-QTL mapping from genotypes in memory"""

        self.complete_tasks = 0

        if logger is None:
            logger = SimpleLogger(verbose=verbose)
        assert np.all(phenotype_df.columns == covariates_df.index)

        variant_ids = genotype_df.index.tolist()
        n_variants = len(variant_ids)
        n_samples = phenotype_df.shape[1]

        logger.write('trans-QTL mapping')
        logger.write('  * {} samples'.format(n_samples))
        logger.write('  * ={} phenotypes'.format(phenotype_df.shape[0]))
        logger.write('  * {} covariates'.format(covariates_df.shape[1]))
        logger.write('  * {} variants'.format(n_variants))
        if interaction_s is not None:
            logger.write('  * including interaction term')

        # # with tf.device('/cpu:0'):
        # Changed from float32 to float32
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df.values,
                                                batch_size=batch_size,
                                                dtype=np.float32)

        # Changed from float32 to float32
        dataset_genotypes = tf.data.Dataset.from_generator(ggt.generate_data,
                                                           output_types=tf.float16)
        dataset_genotypes = dataset_genotypes.prefetch(10)
        iterator = dataset_genotypes.make_one_shot_iterator()
        next_element = iterator.get_next()
        #
        # # index of VCF samples corresponding to phenotypes
        ix_t = get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)

        _ix_t = _get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = INTRAOP
        config.inter_op_parallelism_threads = INTEROP

        self.sess = tf.Session(config=config)

        self.next_element = tf.gather(next_element, ix_t, axis=1)

    def fetch_g_iter(self):
        return self.sess.run(self.next_element)

    def send_task_complete_message(self):
        self.complete_tasks += 1

    def fetch_complete_tasks(self):
        return self.complete_tasks


def map_trans(genotype_df, phenotype_df, covariates_df,
              interaction_s=None,
              return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
              return_r2=False, batch_size=20000, logger=None, verbose=True):
    """Run trans-QTL mapping from genotypes in memory"""

    ps = ParameterServer.remote(genotype_df, phenotype_df, covariates_df,
                                interaction_s=interaction_s,
                                return_sparse=return_sparse,
                                pval_threshold=pval_threshold,
                                maf_threshold=maf_threshold,
                                return_r2=return_r2, batch_size=batch_size,
                                logger=logger,
                                verbose=verbose)

    # Changed from float32 to float32
    ggt = genotypeio.GenotypeGeneratorTrans(genotype_df.values,
                                            batch_size=batch_size,
                                            dtype=np.float32)

    # # calculate correlation threshold for sparse output
    # if return_sparse:
    #     dof = n_samples - 2 - covariates_df.shape[1]
    #     tstat_threshold = stats.t.ppf(pval_threshold / 2, dof)
    #     r2_threshold = tstat_threshold ** 2 / (dof + tstat_threshold ** 2)
    # else:
    #     r2_threshold = None
    #     tstat_threshold = None

    # if interaction_s is None:
    #     genotypes, phenotypes, covariates = _initialize_data(phenotype_df,
    #                                                          covariates_df,
    #                                                          batch_size=batch_size,
    #                                                          dtype=tf.float32)
    #
    # else:
    #     genotypes, phenotypes, covariates, interaction = initialize_data(
    #         phenotype_df, covariates_df, batch_size=batch_size,
    #         interaction_s=interaction_s)

    start_time = time.time()
    if verbose:
        print('  Mapping batches')

    pval_list = []
    maf_list = []
    r2_list = []

    max_n_workers = 50

    n_batches = ggt.num_batches  # 336

    n_ray_batches = n_batches // max_n_workers

    if n_batches % max_n_workers != 0:
        n_ray_batches += 1

    if max_n_workers > n_batches:
        max_n_workers = n_batches

    for ray_batch in range(n_ray_batches):
        workers = [
            worker_task.remote(ps, phenotype_df, covariates_df, interaction_s,
                               batch_size, return_sparse, pval_threshold,
                               return_r2)
            for _ in range(max_n_workers)]

        for i in tqdm.tqdm(range(max_n_workers),
                           desc='Processing batch {}/{}'.format(i,
                                                                n_ray_batches)):
            while True:
                time.sleep(1e-1)
                num_comlete_tasks = ps.fetch_complete_taks.remote() % \
                                    max_n_workers
                if num_comlete_tasks >= i + 1:
                    break

    # workers =  [worker_task.remote(ps, phenotype_df, covariates_df,
    # interaction_s,
    #         batch_size, return_sparse,pval_threshold,return_r2)
    #         for _ in range(ggt.num_batches)]
    #
    # for i in tqdm.tqdm(range(ggt.num_batches),desc='Processing batches'):
    #     while True:
    #         time.sleep(1e-1)
    #         num_comlete_tasks = int(ray.get(ps.fetch_complete_tasks.remote()))
    #         if num_comlete_tasks >= i+1:
    #             break

    print(data)
    print("Reached the end!")

    # if return_sparse:
    #     pval = tf.sparse_concat(0, pval_list).eval()
    # else:
    #     pval = tf.concat(pval_list, 0).eval()
    # maf = tf.concat(maf_list, 0).eval()
    # if return_r2:
    #     r2 = tf.concat(r2_list, 0).eval()
    # if verbose:
    #     print()  # writer.close()
    #
    # logger.write(
    #     '    time elapsed: {:.2f} min'.format(
    #         (time.time() - start_time) / 60))
    #
    # if return_sparse:
    #     ix = pval.indices[:, 0] < n_variants  # truncate last batch
    #     v = [variant_dict[i] for i in pval.indices[ix, 0]]
    #     if phenotype_df.shape[0] > 1:
    #         phenotype_ids = phenotype_df.index[pval.indices[ix, 1]]
    #     else:
    #         phenotype_ids = phenotype_df.index.tolist() * len(pval.values)
    #     pval_df = pd.DataFrame(
    #         np.array([v, phenotype_ids, pval.values[ix], maf[ix]]).T,
    #         columns=['variant_id', 'phenotype_id', 'pval', 'maf'])
    #
    #     pval_df['pval'] = pval_df['pval'].astype(np.float64)
    #     pval_df['maf'] = pval_df['maf'].astype(np.float32)
    #     if return_r2:
    #         pval_df['r2'] = r2[ix].astype(np.float32)
    # else:
    #     # truncate last batch
    #     pval = pval[:n_variants]
    #     maf = maf[:n_variants]
    #     # add indices
    #     pval_df = pd.DataFrame(pval, index=variant_ids,
    #                            columns=[i for i in phenotype_df.index])
    #     pval_df['maf'] = maf
    #     pval_df.index.name = 'variant_id'
    #
    # if maf_threshold is not None and maf_threshold > 0:
    #     logger.write(
    #         '  * filtering output by MAF >= {}'.format(maf_threshold))
    #     pval_df = pval_df[pval_df['maf'] >= maf_threshold]
    #
    # logger.write('done.')
    # return pval_df

    def fetch_g_iter(self):
        return self.sess.run(self.next_element)


"""
==============================================================================================================================================================
==============================================================================================================================================================

==============================================================================================================================================================
==============================================================================================================================================================
"""


def map_trans_permutations(genotype_input, phenotype_df, covariates_df,
                           permutations=None, split_chr=True, pval_df=None,
                           nperms=10000, maf_threshold=0.05, batch_size=20000,
                           logger=None):
    """
    Warning: this function requires that all phenotypes are normally
    distributed,
             e.g., inverse normal transformed
    """
    if logger is None:
        logger = SimpleLogger()
    assert np.all(phenotype_df.columns == covariates_df.index)

    if split_chr:
        plink_reader = genotype_input
        # assert isinstance(plink_reader, genotypeio.PlinkReader)
        if pval_df is not None:
            assert 'phenotype_chr' in pval_df and 'pval' in pval_df and np.all(
                pval_df.index == phenotype_df.index)
        variant_ids = plink_reader.bim['snp'].tolist()
        # index of VCF samples corresponding to phenotypes
        ix_t = get_sample_indexes(plink_reader.fam['iid'].tolist(),
                                  phenotype_df)
    else:
        genotype_df = genotype_input
        assert isinstance(genotype_df, pd.DataFrame)
        variant_ids = genotype_df.index.tolist()
        # index of VCF samples corresponding to phenotypes
        ix_t = get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)

    n_variants = len(variant_ids)
    n_samples = phenotype_df.shape[1]
    dof = phenotype_df.shape[1] - 2 - covariates_df.shape[1]

    logger.write('trans-QTL mapping (FDR)')
    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))

    if permutations is None:  # generate permutations
        q = stats.norm.ppf(np.arange(1, n_samples + 1) / (n_samples + 1))
        qv = np.tile(q, [nperms, 1])
        for i in np.arange(nperms):
            np.random.shuffle(qv[i, :])
    else:
        assert permutations.shape[1] == n_samples
        nperms = permutations.shape[0]
        qv = permutations
        logger.write('  * {} permutations'.format(nperms))
    permutations_t = tf.constant(qv, dtype=tf.float32)
    permutations_t = tf.reshape(permutations_t, shape=[-1, n_samples])

    genotypes_t, phenotypes_t, covariates_t = initialize_data(phenotype_df,
                                                              covariates_df,
                                                              batch_size=batch_size,
                                                              dtype=tf.float32)
    max_r2_nominal_t, max_r2_permuted_t, idxmax_t = _calculate_max_r2(
        genotypes_t, phenotypes_t, permutations_t, covariates_t,
        maf_threshold=maf_threshold)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    if split_chr:
        start_time = time.time()
        chr_max_r2_nominal = OrderedDict()
        chr_max_r2_empirical = OrderedDict()
        print('  Mapping batches')
        with tf.Session() as sess:
            sess.run(init_op)

            for chrom in plink_reader.bim['chrom'].unique():
                genotypes, variant_pos = plink_reader.get_region(chrom,
                                                                 verbose=True)
                ggt = genotypeio.GenotypeGeneratorTrans(genotypes,
                                                        batch_size=batch_size,
                                                        dtype=np.float32)
                dataset_genotypes = tf.data.Dataset.from_generator(
                    ggt.generate_data, output_types=tf.float32)
                dataset_genotypes = dataset_genotypes.prefetch(10)
                iterator = dataset_genotypes.make_one_shot_iterator()
                next_element = iterator.get_next()
                next_element = tf.gather(next_element, ix_t, axis=1)

                nominal_list = []
                perms_list = []
                nominal_idx_list = []
                for i in range(1, ggt.num_batches + 1):
                    sys.stdout.write(
                        '\r  * {}: processing batch {}/{}  '.format(chrom, i,
                                                                    ggt.num_batches))
                    sys.stdout.flush()

                    g_iter = sess.run(next_element)
                    max_r2_nominal, max_r2_permuted = sess.run(
                        [max_r2_nominal_t, max_r2_permuted_t],
                        feed_dict={genotypes_t: g_iter})
                    nominal_list.append(max_r2_nominal)
                    perms_list.append(max_r2_permuted)

                chr_max_r2_nominal[chrom] = np.max(np.array(nominal_list), 0)
                chr_max_r2_empirical[chrom] = np.max(np.array(perms_list), 0)
        logger.write('    time elapsed: {:.2f} min'.format(
            (time.time() - start_time) / 60))

        chr_max_r2_nominal = pd.DataFrame(chr_max_r2_nominal,
                                          index=phenotype_df.index)
        chr_max_r2_empirical = pd.DataFrame(chr_max_r2_empirical)

        # compute leave-one-out max
        max_r2_nominal = OrderedDict()
        max_r2_empirical = OrderedDict()
        for c in chr_max_r2_nominal:
            max_r2_nominal[c] = chr_max_r2_nominal[
                np.setdiff1d(chr_max_r2_nominal.columns, c)].max(1)
            max_r2_empirical[c] = chr_max_r2_empirical[
                np.setdiff1d(chr_max_r2_empirical.columns, c)].max(1)
        max_r2_nominal = pd.DataFrame(max_r2_nominal, index=phenotype_df.index)
        max_r2_empirical = pd.DataFrame(max_r2_empirical)  # nperms x chrs

        if pval_df is not None:
            # nominal p-value (sanity check, matches pval_df['pval'])
            r2_nominal = max_r2_nominal.lookup(pval_df['phenotype_id'],
                                               pval_df['phenotype_chr'])
            tstat = np.sqrt(dof * r2_nominal / (1 - r2_nominal))
            minp_nominal = pd.Series(2 * stats.t.cdf(-np.abs(tstat), dof),
                                     index=pval_df['phenotype_id'])

        # empirical p-values
        tstat = np.sqrt(dof * max_r2_empirical / (1 - max_r2_empirical))
        minp_empirical = pd.DataFrame(2 * stats.t.cdf(-np.abs(tstat), dof),
                                      columns=tstat.columns)
        if pval_df is not None:
            pval_perm = np.array(
                [(np.sum(minp_empirical[chrom] <= p) + 1) / (nperms + 1) for
                    p, chrom in zip(pval_df['pval'], pval_df['phenotype_chr'])])

        beta_shape1 = {}
        beta_shape2 = {}
        true_dof = {}
        for c in max_r2_empirical:
            beta_shape1[c], beta_shape2[c], true_dof[c] = fit_beta_parameters(
                max_r2_empirical[c], dof)
        if pval_df is not None:
            beta_shape1 = [beta_shape1[c] for c in pval_df['phenotype_chr']]
            beta_shape2 = [beta_shape2[c] for c in pval_df['phenotype_chr']]
            true_dof = [true_dof[c] for c in pval_df['phenotype_chr']]
        else:
            chroms = plink_reader.bim['chrom'].unique()
            beta_shape1 = [beta_shape1[c] for c in chroms]
            beta_shape2 = [beta_shape2[c] for c in chroms]
            true_dof = [true_dof[c] for c in chroms]

        if pval_df is not None:
            pval_true_dof = pval_from_corr(r2_nominal, true_dof)
            pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)
            variant_id = [np.NaN] * len(pval_beta)

    else:
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df.values,
                                                batch_size=batch_size,
                                                dtype=np.float32)
        dataset_genotypes = tf.data.Dataset.from_generator(ggt.generate_data,
                                                           output_types=tf.float32)
        dataset_genotypes = dataset_genotypes.prefetch(10)
        iterator = dataset_genotypes.make_one_shot_iterator()
        next_element = iterator.get_next()
        next_element = tf.gather(next_element, ix_t, axis=1)

        start_time = time.time()
        max_r2_nominal = []
        max_r2_nominal_idx = []
        max_r2_empirical = []
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(ggt.num_batches):
                sys.stdout.write(
                    '\rProcessing batch {}/{}'.format(i + 1, ggt.num_batches))
                sys.stdout.flush()
                g_iter = sess.run(next_element)
                res = sess.run([max_r2_nominal_t, max_r2_permuted_t, idxmax_t],
                               feed_dict={genotypes_t: g_iter})
                max_r2_nominal.append(res[0])
                max_r2_nominal_idx.append(res[2] + i * batch_size)
                max_r2_empirical.append(res[1])
            print()
        max_r2_nominal = np.array(max_r2_nominal)
        max_r2_nominal_idx = np.array(max_r2_nominal_idx)
        max_r2_empirical = np.array(max_r2_empirical)
        logger.write('    time elapsed: {:.2f} min'.format(
            (time.time() - start_time) / 60))

        if len(max_r2_nominal_idx.shape) == 1:
            max_r2_nominal = max_r2_nominal.reshape(-1, 1)
            max_r2_nominal_idx = max_r2_nominal_idx.reshape(-1, 1)

        idxmax = np.argmax(max_r2_nominal, 0)
        variant_ix = [max_r2_nominal_idx[i, k] for k, i in enumerate(idxmax)]
        variant_id = genotype_df.index[variant_ix]

        max_r2_nominal = np.max(max_r2_nominal, 0)
        tstat = np.sqrt(dof * max_r2_nominal / (1 - max_r2_nominal))
        minp_nominal = 2 * stats.t.cdf(-np.abs(tstat), dof)

        max_r2_empirical = np.max(max_r2_empirical, 0)
        tstat = np.sqrt(dof * max_r2_empirical / (1 - max_r2_empirical))
        minp_empirical = 2 * stats.t.cdf(-np.abs(tstat), dof)

        pval_perm = np.array(
            [(np.sum(minp_empirical <= p) + 1) / (nperms + 1) for p in
                minp_nominal])
        # calculate beta p-values for each phenotype:
        beta_shape1, beta_shape2, true_dof, minp_vec = fit_beta_parameters(
            max_r2_empirical, dof, tol=1e-4, return_minp=True)
        pval_true_dof = pval_from_corr(max_r2_nominal, true_dof)
        pval_beta = stats.beta.cdf(pval_true_dof, beta_shape1, beta_shape2)

    if not split_chr or pval_df is not None:
        fit_df = pd.DataFrame(OrderedDict(
            [('beta_shape1', beta_shape1), ('beta_shape2', beta_shape2),
                ('true_df', true_dof), ('min_pval_true_df', pval_true_dof),
                ('variant_id', variant_id), ('min_pval_nominal', minp_nominal),
                ('pval_perm', pval_perm), ('pval_beta', pval_beta), ]),
            index=phenotype_df.index)
    else:
        fit_df = pd.DataFrame(OrderedDict(
            [('beta_shape1', beta_shape1), ('beta_shape2', beta_shape2),
                ('true_df', true_dof), ]), index=chroms)

    if split_chr:
        return fit_df, minp_empirical
    else:
        return fit_df, minp_vec


def map_trans_tfrecord(vcf_tfrecord, phenotype_df, covariates_df,
                       interaction_s=None, return_sparse=True,
                       pval_threshold=1e-5, maf_threshold=0.05,
                       batch_size=50000, logger=None):
    """Run trans-QTL mapping from genotypes in tfrecord"""
    if logger is None:
        logger = SimpleLogger()
    assert np.all(phenotype_df.columns == covariates_df.index)

    with open(vcf_tfrecord + '.samples') as f:
        vcf_sample_ids = f.read().strip().split('\n')
    n_samples_vcf = len(vcf_sample_ids)

    with gzip.open(vcf_tfrecord + '.variants.gz', 'rt') as f:
        variant_ids = f.read().strip().split('\n')
    variant_dict = {i: j for i, j in enumerate(variant_ids)}
    n_variants = len(variant_ids)

    # index of VCF samples corresponding to phenotypes
    ix_t = get_sample_indexes(vcf_sample_ids, phenotype_df)
    n_samples = phenotype_df.shape[1]

    # batched_dataset = dataset.apply(tf.contrib.data.padded_batch(
    # batch_size, padded_shapes=[[batch_size], [None]]))
    # batched_dataset = dataset.padded_batch(batch_size, padded_shapes=(
    # batch_size,n_samples), padding_values=0)
    with tf.device('/cpu:0'):
        batched_dataset = genotypeio.make_batched_dataset(vcf_tfrecord,
                                                          batch_size, ix_t=ix_t)

        iterator = batched_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        next_element = genotypeio.pad_up_to(next_element, [batch_size,
            n_samples])  # not clear if right/best way to do this

    logger.write('  * {} samples'.format(n_samples))
    logger.write('  * {} phenotypes'.format(phenotype_df.shape[0]))
    logger.write('  * {} covariates'.format(covariates_df.shape[1]))
    logger.write('  * {} variants'.format(n_variants))
    if interaction_s is not None:
        logger.write('  * including interaction term')

    num_batches = int(np.ceil(np.true_divide(n_variants, batch_size)))

    # calculate correlation threshold for sparse output
    if return_sparse:
        dof = n_samples - 2 - covariates_df.shape[1]
        t = stats.t.ppf(pval_threshold / 2, dof) ** 2 / dof
        r2_threshold = t / (1 + t)
    else:
        r2_threshold = None

    if interaction_s is None:
        genotypes, phenotypes, covariates = initialize_data(phenotype_df,
                                                            covariates_df,
                                                            batch_size=batch_size)
        with tf.device('/gpu:0'):
            p_values, maf = calculate_association(genotypes, phenotypes,
                                                  covariates,
                                                  return_sparse=return_sparse,
                                                  r2_threshold=r2_threshold)
    else:
        genotypes, phenotypes, covariates, interaction = initialize_data(
            phenotype_df, covariates_df, batch_size=batch_size,
            interaction_s=interaction_s)
        p_values, maf = calculate_association(genotypes, phenotypes, covariates,
                                              interaction_t=interaction,
                                              return_sparse=return_sparse,
                                              r2_threshold=r2_threshold)

    # g = _parse_function(next_element, batch_size, n_samples, ix_t)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    start_time = time.time()
    with tf.Session() as sess:
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        # writer = tf.summary.FileWriter('logs', sess.graph, session=sess)
        sess.run(init_op)
        pval_list = []
        maf_list = []
        for i in range(num_batches):
            sys.stdout.write(
                '\rProcessing batch {}/{}'.format(i + 1, num_batches))
            sys.stdout.flush()

            g_iter = sess.run(next_element)
            # g_iter = sess.run(g)
            p_ = sess.run([p_values, maf], feed_dict={genotypes: g_iter})  # ,
            # options=run_options, run_metadata=run_metadata)
            # writer.add_run_metadata(run_metadata, 'batch%d' % i)

            pval_list.append(p_[0])
            maf_list.append(p_[1])

        if return_sparse:
            pval = tf.sparse_concat(0, pval_list).eval()
        else:
            pval = tf.concat(pval_list, 0).eval()
        maf = tf.concat(maf_list, 0).eval()
        print()  # writer.close()

    logger.write(
        'Time elapsed: {:.2f} min'.format((time.time() - start_time) / 60))

    if return_sparse:
        ix = pval.indices[:, 0] < n_variants  # truncate last batch
        v = [variant_dict[i] for i in pval.indices[ix, 0]]
        pval_df = pd.DataFrame(np.array(
            [v, phenotype_df.index[pval.indices[ix, 1]], pval.values[ix],
                maf[ix]]).T,
                               columns=['variant_id', 'phenotype_id', 'pval',
                                   'maf'])
        pval_df['pval'] = pval_df['pval'].astype(np.float64)
        pval_df['maf'] = pval_df['maf'].astype(np.float32)
    else:
        # truncate last batch
        pval = pval[:n_variants]
        maf = maf[:n_variants]
        # add indices
        pval_df = pd.DataFrame(pval, index=variant_ids,
                               columns=[i for i in phenotype_df.index])
        pval_df['maf'] = maf
        pval_df.index.name = 'variant_id'

    if maf_threshold is not None and maf_threshold > 0:
        logger.write('  * filtering output by MAF >= {}'.format(maf_threshold))
        pval_df = pval_df[pval_df['maf'] >= maf_threshold]

    return pval_df


def _in_cis(chrom, pos, gene_id, tss_dict, window=1000000):
    """Test if a variant-gene pair is in cis"""
    if chrom == tss_dict[gene_id]['chr']:
        tss = tss_dict[gene_id]['tss']
        if pos >= tss - window and pos <= tss + window:
            return True
        else:
            return False
    else:
        return False


def filter_cis(pval_df, tss_dict, window=1000000):
    """Filter out cis-QTLs"""
    drop_ix = []
    for k, gene_id, variant_id in zip(pval_df['phenotype_id'].index,
                                      pval_df['phenotype_id'],
                                      pval_df['variant_id']):
        chrom, pos = variant_id.split('_', 2)[:2]
        pos = int(pos)
        if _in_cis(chrom, pos, gene_id, tss_dict, window=window):
            drop_ix.append(k)
    return pval_df.drop(drop_ix)


# ------------------------------------------------------------------------------
#  Input parsers
# ------------------------------------------------------------------------------
def read_phenotype_bed(phenotype_bed):
    """Load phenotype BED file as phenotype and TSS DataFrames"""
    if phenotype_bed.endswith('.bed.gz'):
        phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3,
                                   dtype={'#chr': str, '#Chr': str})
    elif phenotype_bed.endswith('.parquet'):
        phenotype_df = pd.read_parquet(phenotype_bed)
    else:
        raise ValueError('Unsupported file type.')
    phenotype_df = phenotype_df.rename(
        columns={i: i.lower() for i in phenotype_df.columns[:3]})
    phenotype_pos_df = phenotype_df[['#chr', 'end']].rename(
        columns={'#chr': 'chr', 'end': 'tss'})
    phenotype_df = phenotype_df.drop(['#chr', 'start', 'end'], axis=1)
    return phenotype_df, phenotype_pos_df


def main():
    parser = argparse.ArgumentParser(
        description='tensorQTL: GPU-based QTL mapper')
    parser.add_argument('genotype_path',
                        help='Genotypes/dosages in PLINK or tfrecord format')
    parser.add_argument('phenotype_bed', help='Phenotypes in BED format')
    parser.add_argument('prefix', help='Prefix for output file names')
    parser.add_argument('--mode', default='cis',
                        choices=['cis', 'cis_nominal', 'cis_independent',
                            'trans'], help='Mapping mode. Default: cis')
    parser.add_argument('--covariates', default=None,
                        help='Covariates file, tab-delimited, covariates x '
                             'samples')
    parser.add_argument('--permutations', default=10000,
                        help='Number of permutations. Default: 10000')
    parser.add_argument('--interaction', default=None, type=str,
                        help='Interaction term')
    parser.add_argument('--cis_output', default=None, type=str,
                        help="Output from 'cis' mode with q-values. Required "
                             "for independent cis-QTL mapping.")
    parser.add_argument('--phenotype_groups', default=None, type=str,
                        help='Phenotype groups. Header-less TSV with two '
                             'columns: phenotype_id, group_id')
    parser.add_argument('--window', default=1000000, type=np.int32,
                        help='Cis-window size, in bases. Default: 1000000.')
    parser.add_argument('--pval_threshold', default=None, type=np.float64,
                        help='Output only significant phenotype-variant pairs '
                             'with a p-value below threshold. Default: 1e-5 '
                             'for trans-QTL')
    parser.add_argument('--maf_threshold', default=None, type=np.float64,
                        help='Include only genotypes with minor allele '
                             'frequency >=maf_threshold. Default: 0')
    parser.add_argument('--maf_threshold_interaction', default=0.05,
                        type=np.float64,
                        help='MAF threshold for interactions, applied to '
                             'lower and upper half of samples')
    parser.add_argument('--return_dense', action='store_true',
                        help='Return dense output for trans-QTL.')
    parser.add_argument('--return_r2', action='store_true',
                        help='Return r2 (only for sparse trans-QTL output)')
    parser.add_argument('--best_only', action='store_true',
                        help='Produce output only for the top '
                             'association/phenotype')
    parser.add_argument('--output_text', action='store_true',
                        help='Write output in txt.gz format instead of '
                             'parquet (trans-QTL mode only)')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Batch size. Reduce this if encountering OOM '
                             'errors.')
    parser.add_argument('--fdr', default=0.05, type=np.float64,
                        help='FDR for cis-QTLs')
    parser.add_argument('--qvalue_lambda', default=None, type=np.float64,
                        help='lambda parameter for pi0est in qvalue.')
    parser.add_argument('-o', '--output_dir', default='.',
                        help='Output directory')
    args = parser.parse_args()

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # check inputs
    if args.mode == 'cis_independent' and (
            args.cis_output is None or not os.path.exists(args.cis_output)):
        raise ValueError("Output from 'cis' mode must be provided.")

    logger = SimpleLogger(os.path.join(args.output_dir,
                                       args.prefix + '.tensorQTL.{}.log'.format(
                                           args.mode)))
    logger.write('[{}] Running TensorQTL: {}-QTL mapping'.format(
        datetime.now().strftime("%b %d %H:%M:%S"), args.mode.split('_')[0]))

    # load inputs
    logger.write('  * reading phenotypes ({})'.format(args.phenotype_bed))
    phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotype_bed)

    tss_dict = phenotype_pos_df.T.to_dict()
    if args.covariates is not None:
        logger.write('  * reading covariates ({})'.format(args.covariates))
        covariates_df = pd.read_csv(args.covariates, sep='\t', index_col=0).T
        assert np.all(phenotype_df.columns == covariates_df.index)
    if args.interaction is not None:
        logger.write(
            '  * reading interaction term ({})'.format(args.interaction))
        interaction_s = pd.read_csv(args.interaction, sep='\t', index_col=0,
                                    header=None, squeeze=True)
        assert np.all(interaction_s.index == covariates_df.index)
    else:
        interaction_s = None

    if args.maf_threshold is None:
        if args.mode == 'trans':
            maf_threshold = 0.05
        else:
            maf_threshold = 0
    else:
        maf_threshold = args.maf_threshold

    if args.phenotype_groups is not None:
        group_s = pd.read_csv(args.phenotype_groups, sep='\t', index_col=0,
                              header=None, squeeze=True)
        # verify sort order
        group_dict = group_s.to_dict()
        previous_group = ''
        parsed_groups = 0
        for i in phenotype_df.index:
            if group_dict[i] != previous_group:
                parsed_groups += 1
                previous_group = group_dict[i]
        if not parsed_groups == len(group_s.unique()):
            raise ValueError(
                'Groups defined in input do not match phenotype file (check '
                'sort order).')
    else:
        group_s = None

    if args.mode.startswith('cis'):
        pr = genotypeio.PlinkReader(args.genotype_path,
                                    select_samples=phenotype_df.columns,
                                    dtype=np.int8)
        if args.mode == 'cis':
            res_df = map_cis(pr, phenotype_df, phenotype_pos_df, covariates_df,
                             group_s=group_s, nperm=args.permutations,
                             logger=logger)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir,
                                    args.prefix + '.cis_qtl.txt.gz')
            if has_rpy2 and group_s is None:
                calculate_qvalues(res_df, fdr=args.fdr,
                                  qvalue_lambda=args.qvalue_lambda)
            res_df.to_csv(out_file, sep='\t', float_format='%.6g',
                          compression='gzip')
        elif args.mode == 'cis_nominal':
            if interaction_s is None:
                map_cis_nominal(pr, phenotype_df, phenotype_pos_df,
                                covariates_df, args.prefix,
                                output_dir=args.output_dir, logger=logger)
            else:
                map_cis_interaction_nominal(pr, phenotype_df, phenotype_pos_df,
                                            covariates_df, interaction_s,
                                            args.prefix,
                                            maf_threshold_interaction=args.maf_threshold_interaction,
                                            best_only=args.best_only,
                                            output_dir=args.output_dir,
                                            logger=logger)
        elif args.mode == 'cis_independent':
            summary_df = pd.read_csv(args.cis_output, sep='\t', index_col=0)
            summary_df.rename(columns={'minor_allele_samples': 'ma_samples',
                'minor_allele_count'                         : 'ma_count'},
                              inplace=True)
            res_df = map_cis_independent(pr, summary_df, phenotype_df,
                                         phenotype_pos_df, covariates_df,
                                         fdr=args.fdr, nperm=args.permutations,
                                         logger=logger)
            logger.write('  * writing output')
            out_file = os.path.join(args.output_dir,
                                    args.prefix + '.cis_independent_qtl.txt.gz')
            res_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g',
                          compression='gzip')
    elif args.mode == 'trans':
        return_sparse = not args.return_dense
        pval_threshold = args.pval_threshold
        if pval_threshold is None and return_sparse:
            pval_threshold = 1e-5
            logger.write('  * p-value threshold: {:.2g}'.format(pval_threshold))

        genotype_df = genotypeio.load_genotypes(args.genotype_path)
        pval_df = map_trans(genotype_df, phenotype_df, covariates_df,
                            interaction_s=interaction_s,
                            return_sparse=return_sparse,
                            pval_threshold=pval_threshold,
                            maf_threshold=maf_threshold,
                            batch_size=args.batch_size,
                            return_r2=args.return_r2, logger=logger)

        logger.write('  * filtering out cis-QTLs (within +/-5Mb)')
        pval_df = filter_cis(pval_df, tss_dict, window=5000000)

        logger.write('  * writing output')
        if not args.output_text:
            pval_df.to_parquet(os.path.join(args.output_dir,
                                            args.prefix +
                                            '.trans_qtl_pairs.parquet'))
        else:
            out_file = os.path.join(args.output_dir,
                                    args.prefix + '.trans_qtl_pairs.txt.gz')
            pval_df.to_csv(out_file, sep='\t', index=False, float_format='%.6g',
                           compression='gzip')

    logger.write('[{}] Finished mapping'.format(
        datetime.now().strftime("%b %d %H:%M:%S")))


if __name__ == '__main__':
    main()
