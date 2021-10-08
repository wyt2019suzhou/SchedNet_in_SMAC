import multiprocessing
import os
import os.path as osp
import sys

import setproctitle
import tensorflow as tf

from alg_parameters import *
from learner import learn
from utilize import get_session, save_state, set_global_seeds
from wrap_env import make_train_env



def train():
    setproctitle.setproctitle('StarCraft-Schednet' + EXPERIMENT_NAME + "@" + USER_NAME)
    env = build_env()
    model = learn(env=env)
    return model, env

def build_env():
    """Build multiple processing environment.
    """
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = N_ENVS or ncpu
    env = make_train_env(nenv)
    return env

def main():
    # set tf environment
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    set_global_seeds(SEED)

    # key function
    model, env = train()

    savepath = osp.join("my_model/", 'final')
    os.makedirs(savepath, exist_ok=True)
    savepath = osp.join(savepath, 'ppomodel')
    save_state(savepath)

    env.close()
    return model


if __name__ == '__main__':
    main()
