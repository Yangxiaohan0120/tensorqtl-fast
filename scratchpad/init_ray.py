#!/bin/env/python
# -*- encoding: utf-8 -*-
"""
Ray helper functions
"""
from __future__ import division, print_function

import os
import subprocess
import time

import ray

ray_args = {

    "redis_address"           : ("--redis-address", str),
    "num_cpus"                : ("--num-cpus", int),
    "num_gpus"                : ("--num-gpus ", int),
    "resources"               : ("--resources", dict),
    "object_store_memory"     : ("--object-store-memory", int),
    "redis_max_memory"        : ("--redis-max-memory", int),
    "log_to_driver"           : (None, bool),
    "node_ip_address"         : (None, str),
    "object_id_seed"          : (None, int),
    "local_mode"              : (None, bool),
    "ignore_reinit_error"     : (None, bool),
    "num_redis_shards"        : ("--num-redis-shards", int),
    "redis_max_clients"       : ("--redis-max-clients", int),
    "redis_password"          : ("--redis-password", str),
    "plasma_directory"        : ("--plasma-directory", str),
    "huge_pages"              : ("--huge-pages", bool),
    "include_webui"           : (None, bool),
    "driver_id"               : (None, str),
    "configure_logging"       : (None, bool),
    "logging_level"           : (None, None),
    "logging_format"          : (None, None),
    "plasma_store_socket_name": ("--plasma-store-socket-name", str),
    "raylet_socket_name"      : ("--raylet-socket-name", str),
    "temp_dir"                : ("--temp-dir", str),
    "load_code_from_local"    : (None, None),
    "_internal_config"        : ("--internal-config", str)
}

# Script for creating new workers
create_worker = """
ray start --redis-address $addr
sleep 3600
"""


def get_ip_addresses(n_workers: int, wait: bool = True) -> set:
    """
    Get the set of connected ip addresses in a ray compute clåuster

    :param n_workers: Number of ray workers
    :param wait: Whether to wait for the whole set of address or return
        the set of currently connected
    """

    @ray.remote
    def f():
        time.sleep(0.01)
        return ray.services.get_node_ip_address()

    if wait:
        while True:
            ips = set(ray.get([f.remote() for _ in range(1000)]))
            time.sleep(5e-1)
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
    arg_names = ['cluster_name', 'num_workers']
    args = [cluster_name, num_workers]
    for var, _type in zip(args, (str, int)):
        assert isinstance(var, _type), (
            "Invalid {} arg type: {}".format(
                arg_names[args.index(var)],
                type(var).__name__
            ))

    if RUN_CLUSTER:

        ray_head_args = []
        for n in kwargs.keys():
            if n in ray_args:
                req_type = ray_args[n][1]
                if req_type != None:
                    if not isinstance(kwargs[n], req_type):
                        raise TypeError(
                            "{} arg must be type {}, not {}".format(
                                n, req_type, type(n).__name__))
                ray_head_args.append(
                    '{}={}'.format(ray_args[n][0], kwargs[n]))
            else:
                raise UserWarning(
                    "Invalid kwarg arg: {}, not in {}".format(n,
                                                              list(
                                                                  ray_args.keys())))

        if cluster_name.upper() == 'VLAB':
            qsub_non_excl = '/opt/pbs/default/bin/qsub'
            qsub_excl = '/opt/pbs/bin/qsub'

        if cluster_name.upper() == 'DEVCLOUD':
            qsub_non_excl = '/usr/local/bin/qsub'
            qsub_excl = qsub_non_excl

        with open('create_worker', 'w') as f:
            f.write(create_worker)

        if cluster_name.upper() in ('VLAB', 'DEVCLOUD'):

            ray_head_cmd = ['ray', 'start', '--head'] + ray_head_args

            print(ray_head_cmd)

            result = subprocess.run(ray_head_cmd,
                                    stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)

            if not result.returncode:
                output = result.stderr
                output = [n for n in output.decode().split('\n')
                    if ' --redis-address' in n]

                worker_cmd = output[0][4:]
                head_ip_addr = worker_cmd.split(' ')[-1]

                ray.init(redis_address=head_ip_addr)

                for i in range(num_workers):

                    worker_cmd = '{} -lselect=1 -lplace=excl ' \
                                 'create_worker -v ' \
                                 'addr={}'.format(
                        qsub_non_excl, head_ip_addr)

                    status = os.system(worker_cmd)

                    if status != 0:
                        worker_cmd = '{} -lselect=1 -lplace=excl ' \
                                     'create_worker -v addr={}'.format(
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
            raise UserWarning(
                "Invalid cluster name: {}".format(cluster_name))
    else:
        ray.init()


if __name__ == "__main__":
    init_ray(redis_max_memory=100000000000, object_store_memory=500000000000)

    while True:
        print(get_ip_addresses(5, wait=False))
        time.sleep(3)

