import subprocess
import ray
import sys
import os

create_worker = """
ray start --redis-address $2 
sleep 3600
"""


def get_ip_addresses(n_workers) -> set:
    """
    Get the set of connected ip addresses in a ray compute cluster
    """
    @ray.remote
    def f():
        time.sleep(0.01)
        return ray.services.get_node_ip_address()

    while True:
        ips = set(ray.get([f.remote() for _ in range(1000)]))
        time.sleep(1e-2)
        if len(ips) >= n_workers + 1:
            break

    return ips


def init_ray(num_workers=5):
    """
    Initialize a ray compute cluster

    :param num_workers: Number of ray workers, default = 5
    """
    assert isinstance(num_workers,int),(
        "Invalid num_workers arg type: {}".format(
        type(num_workers).__name__
    ))

    ON_DEVCLOUD = False
    ON_VLAB = True
    RUN_CLUSTER = False

    if RUN_CLUSTER:

        if ON_VLAB and ON_DEVCLOUD:
            raise UserWarning("Can be on DevCloud or vLab, not both")

        if ON_VLAB:
            qsub_non_excl = '/opt/pbs/default/bin/qsub'
            qsub_excl = '/opt/pbs/bin/qsub'

        if ON_DEVCLOUD:
            qsub_non_excl = '/usr/local/bin/qsub'
            qsub_excl = qsub_non_excl

        with open('create_worker','w') as f:
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

                ray.init(radis_address=head_ip_addr)

                for i in range(num_workers):

                    worker_cmd = '{} create_worker {}'.format(
                        qsub_non_excl, head_ip_addr)

                    worked = os.system(worker_cmd)

                    if worked != 0:
                        worker_cmd = '{} create_worker {}'.format(
                            qsub_excl, head_ip_addr)

                        worked = os.system(worker_cmd)

                        if worked != 0:
                            raise UserWarning(
                                "Ray unable to generate workers: {}".format(
                                    worker_cmd
                                ))
            else:
                raise Exception("Ray unable to start")
        else:
            ray.init()

def main():
    init_ray()

if __name__ == "__main__":
    main()
