import tensorflow as tf
import dill
import time
import ray


@ray.remote
def worker_task(ps):
    sess = tf.Session()

    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    while True:
        print(ray.get(ps.add.remote(2,2)))
        time.sleep(1)


@ray.remote
class ParameterServer(object):
    def __init__(self):

        self.a = tf.placeholder(tf.int16)
        self.b = tf.placeholder(tf.int16)

        self.sess = tf.Session()

        self.add = tf.add(self.a,self.b)


    def add(self,a,b):
        return self.sess.run(self.add,feed_dict={self.a:a,self.b:b})



def main():

    ray.init()

    # Basic constant operations
    # The value returned by the constructor represents the output
    # of the Constant op.
    ps = ParameterServer.remote()

    ray.get(worker_task.remote(ps))


if __name__ == "__main__":
    main()
