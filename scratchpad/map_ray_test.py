import time

import ray
import tensorflow as tf


@ray.remote
def worker_task(ps):
    g_iter = ps.fetch_g_iter.remote()

    with tf.Session() as sess:
        return sess.run(x, feed_dict={genotypes: g_iter})  # ,
        # options=run_options, run_metadata=run_metadata)
        # writer.add_run_metadata(run_metadata, 'batch{}'.format(i))

        # pval_list.append(p_[0])
        # maf_list.append(p_[1])
        # if return_r2:
        #     r2_list.append(p_[2])


@ray.remote
class map_trans(object):

    def __init__(self, genotype_df, phenotype_df, covariates_df,
                 interaction_s=None,
                 return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
                 return_r2=False, batch_size=20000, logger=None, verbose=True):
        """Run trans-QTL mapping from genotypes in memory"""
        if logger is None:
            logger = SimpleLogger(verbose=verbose)
        assert np.all(phenotype_df.columns == covariates_df.index)

        variant_ids = genotype_df.index.tolist()
        variant_dict = {i: j for i, j in enumerate(variant_ids)}
        n_variants = len(variant_ids)
        n_samples = phenotype_df.shape[1]

        logger.write('trans-QTL mapping')
        logger.write('  * {} samples'.format(n_samples))
        logger.write('  * ={} phenotypes'.format(phenotype_df.shape[0]))
        logger.write('  * {} covariates'.format(covariates_df.shape[1]))
        logger.write('  * {} variants'.format(n_variants))
        if interaction_s is not None:
            logger.write('  * including interaction term')

        # with tf.device('/cpu:0'):
        ggt = genotypeio.GenotypeGeneratorTrans(genotype_df.values,
                                                batch_size=batch_size,
                                                dtype=np.float32)
        dataset_genotypes = tf.data.Dataset.from_generator(ggt.generate_data,
                                                           output_types=tf.float32)
        dataset_genotypes = dataset_genotypes.prefetch(10)
        iterator = dataset_genotypes.make_one_shot_iterator()
        next_element = iterator.get_next()

        # index of VCF samples corresponding to phenotypes
        ix_t = get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)

        _ix_t = _get_sample_indexes(genotype_df.columns.tolist(), phenotype_df)

        print(ix_t)
        self.next_element = tf.gather(next_element, ix_t, axis=1)

        # calculate correlation threshold for sparse output
        if return_sparse:
            dof = n_samples - 2 - covariates_df.shape[1]
            tstat_threshold = stats.t.ppf(pval_threshold / 2, dof)
            r2_threshold = tstat_threshold ** 2 / (dof + tstat_threshold ** 2)
        else:
            r2_threshold = None
            tstat_threshold = None

        if interaction_s is None:
            genotypes, phenotypes, covariates = initialize_data(phenotype_df,
                                                                covariates_df,
                                                                batch_size=batch_size,
                                                                dtype=tf.float32)
            # with tf.device('/gpu:0'):
            x = calculate_association(genotypes, phenotypes, covariates,
                                      return_sparse=return_sparse,
                                      r2_threshold=r2_threshold,
                                      return_r2=return_r2)
        else:
            genotypes, phenotypes, covariates, interaction = initialize_data(
                phenotype_df, covariates_df, batch_size=batch_size,
                interaction_s=interaction_s)
            x = calculate_association(genotypes, phenotypes, covariates,
                                      interaction_t=interaction,
                                      return_sparse=return_sparse,
                                      r2_threshold=r2_threshold)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        start_time = time.time()
        if verbose:
            print('  Mapping batches')
        self.sess = tf.Session()

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # writer = tf.summary.FileWriter('logs', sess.graph, session=sess)

        self.sess.run(init_op)
        pval_list = []
        maf_list = []
        r2_list = []

        data = ray.get([worker_task.remote(self) for _  in range(ggt.num_batches)])

        print(data)

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

@ray.remote
def worker_task(ps):
    while True:
        print(ray.get(ps.add.remote(2, 2)))
        time.sleep(1)


@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.a = tf.placeholder(tf.int16)
        self.b = tf.placeholder(tf.int16)

        self.sess = tf.Session()

        self.add = tf.add(self.a, self.b)

    def add(self, a, b):
        return self.sess.run(self.add, feed_dict={self.a: a, self.b: b})


def main():
    ray.init()

    # Basic constant operations
    # The value returned by the constructor represents the output
    # of the Constant op.
    ps = ParameterServer.remote()

    ray.get(worker_task.remote(ps))


if __name__ == "__main__":
    main()
