import random

import numpy as np
import tensorflow as tf


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
    return sess

def set_global_seeds(i):
    myseed = i if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def save_state(fname):
    saver = tf.train.Saver()
    sess = get_session()
    saver.save(sess, fname)

ALREADY_INITIALIZED = set()

def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

def calculate_neglogp(latent, actions):
    if actions.dtype in {tf.uint8, tf.int32, tf.int64}:
        actions = tf.one_hot(actions, latent.get_shape().as_list()[-1])
    else:
        actions = tf.one_hot(actions, np.shape(latent)[-1])
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=latent,
        labels=actions)

def calculate_entropy(latent):
    a0 = latent - tf.reduce_max(latent, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

def sample(latent):
    u = tf.random_uniform(tf.shape(latent), dtype=latent.dtype)
    return tf.argmax(latent - tf.log(-tf.log(u)), axis=-1)
