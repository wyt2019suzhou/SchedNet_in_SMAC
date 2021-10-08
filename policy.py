import numpy as np
import tensorflow as tf
from net import Net
from alg_parameters import *


class Policy(object):
    """Build action policy of model."""

    def __init__(self, env, batch_size, sess=None):
        # Build  net
        self.batch_size=batch_size
        self.obs = tf.placeholder(shape=(batch_size,) + (env.num_agents,) + (env.obs_shape+1,), dtype=tf.float32,
                                  name='ob')
        self.state = tf.placeholder(shape=(batch_size,) + (env.state_shape+env.num_agents,), dtype=tf.float32,
                                    name='state')
        self.schedule_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, env.num_agents])
        self.priority_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, env.num_agents])

        network = Net()
        self.logits=network.generate_actor_network(self.obs,self.schedule_ph)
        self.ps = tf.nn.softmax(self.logits)
        self.dist = tf.distributions.Categorical(logits=self.logits)
        schedule_policy = network.generate_wg(self.obs)
        self.schedule_policy=tf.squeeze(schedule_policy,axis=-1)
        values, sch_values = network.generate_critic_network(self.state, self.priority_ph)
        self.values = tf.squeeze(values, axis=-1)
        self.sch_values = tf.squeeze(sch_values, axis=-1)
        self.sess = sess



    def step(self, observation, state, schedule_n,priority,valid_action):
        actions = np.zeros((N_ENVS, N_AGENTS))
        feed_dict = {self.obs: observation, self.state: state,self.schedule_ph:schedule_n,self.priority_ph:priority}
        v,sch_values,ps= self.sess.run([self.values,self.sch_values,self.ps], feed_dict)
        ps=np.clip(ps,1e-10,1.0)
        ps[valid_action == 0.0] = 0.0
        ps /= np.expand_dims(np.sum(ps, axis=-1), -1)
        for i in range(N_ENVS):
            for j in range(N_AGENTS):
                actions[i, j] = np.random.choice(range(N_ACTIONS), p=ps[i, j])
        return actions, v,sch_values,ps

    def value(self, state,priority):
        feed_dict = {self.state: state,self.priority_ph:priority}
        v, sch_values=self.sess.run([self.values, self.sch_values], feed_dict)
        return v, sch_values

    def evalue(self, observation, schedule_n,valid_action):
        valid_action = np.array(valid_action, dtype=np.float)
        valid_action = np.expand_dims(valid_action, axis=0)
        feed_dict = {self.obs: observation,self.schedule_ph:schedule_n}
        ps = self.sess.run([self.ps], feed_dict)
        ps=np.squeeze(np.array(ps),0)
        ps[valid_action == 0.0] = 0.0
        evalue_action = np.argmax(ps, axis=-1)

        return evalue_action

    def get_schedule(self,observation):
        feed_dict = {self.obs: observation}
        priority=self.sess.run(self.schedule_policy, feed_dict)
        schedule_idx = np.argsort(-priority)[:,s_num]
        ret = np.zeros((self.batch_size,N_AGENTS))
        for i in range(self.batch_size):
            ret[i,schedule_idx[i]] = 1.0
        return ret, priority

