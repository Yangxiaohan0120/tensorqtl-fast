#!/bin/env/python
#-*- encoding: utf-8 -*-
"""
Ray helper functions
"""
from __future__ import print_function
from __future__ import division
import subprocess
import ray
import os

# Script for creating new workers
create_worker = """
ray start --redis-address $2 
sleep 3600
"""

def get_ip_addresses(n_workers: int, wait: bool = True) -> set:
    """
    Get the set of connected ip addresses in a ray compute cluster
    """

    @ray.remote
    def f():
        time.sleep(0.01)
        return ray.services.get_node_ip_address()

    if wait:
        while True:
            ips = set(ray.get([f.remote() for _ in range(1000)]))
            time.sleep(1e-2)
            if len(ips) >= n_workers + 1:
                break
    else:
        ips = set(ray.get([f.remote() for _ in range(1000)]))

    return ips


def init_ray(num_workers: int = 5, RUN_CLUSTER: bool = True,
             cluster_name: str = 'vLab', **kwargs) -> None:
    """
    Initialize a ray compute cluster

    :param num_workers: Number of ray workers, default = 5
    """
    for var,_type in zip((cluster_name,num_workers),(str,int)):
        assert isinstance(var,_type), (
            "Invalid {} arg type: {}".format(
                type(var).__name__
            ))


    if RUN_CLUSTER:

        if not ON_DEVCLOUD and not ON_VLAB:
            raise UserWarning("Select vLab or DevCloud")


        if cluster_name.upper() == 'VLAB':
            qsub_non_excl = '/opt/pbs/default/bin/qsub'
            qsub_excl = '/opt/pbs/bin/qsub'

        if cluster_name.upper() == 'DEVCLOUD':
            qsub_non_excl = '/usr/local/bin/qsub'
            qsub_excl = qsub_non_excl

        with open('create_worker', 'w') as f:
            f.write(create_worker)

        if ON_VLAB or ON_DEVCLOUD:

            result = subprocess.run(['ray', 'start', '--head'],
                                    stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)

            if not result.returncode:
                output = result.stderr
                output = [n for n in output.decode().split('\n')
                    if ' --redis-address' in n]

                worker_cmd = output[0][4:]
                head_ip_addr = worker_cmd.split(' ')[-1]

                ray.init(radis_address=head_ip_addr,kwargs)

                for i in range(num_workers):

                    worker_cmd = '{} create_worker {}'.format(
                        qsub_non_excl, head_ip_addr)

                    status = os.system(worker_cmd)

                    if status != 0:
                        worker_cmd = '{} create_worker {}'.format(
                            qsub_excl, head_ip_addr)

                        status = os.system(worker_cmd)

                        if status != 0:
                            raise UserWarning(
                                "Ray unable to generate workers: {}".format(
                                    worker_cmd
                                ))
            else:
                raise Exception("Ray unable to start")
        else:
            ray.init()


if __name__ == "__main__":
    init_ray()
