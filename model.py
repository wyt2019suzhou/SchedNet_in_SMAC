import tensorflow as tf

from alg_parameters import *
from policy import Policy
from utilize import get_session, initialize
import numpy as np


class Model(object):
    """Build tensor graph and calculate flow."""
    def __init__(self, env):
        self.sess = get_session()

        with tf.variable_scope('ppo_model', reuse=tf.AUTO_REUSE):
            self.act_model = Policy(env,N_ENVS, self.sess)
            self.train_model=train_model = Policy(env, MINIBATCH_SIZE, self.sess)
            self.evalue_model = Policy(env, 1, self.sess)

        # Create placeholders
        self.action = tf.placeholder(tf.int32, [MINIBATCH_SIZE,N_AGENTS])
        self.advantage = tf.placeholder(tf.float32, [MINIBATCH_SIZE])
        self.returns = tf.placeholder(tf.float32, [MINIBATCH_SIZE])
        self.sch_returns = tf.placeholder(tf.float32, [MINIBATCH_SIZE])
        # Keep track of old actor
        self.old_ps= tf.placeholder(tf.float32, [MINIBATCH_SIZE,N_AGENTS,N_ACTIONS])
        # Keep track of old critic
        self.old_v = tf.placeholder(tf.float32, [MINIBATCH_SIZE])
        self.old_sch_v = tf.placeholder(tf.float32, [MINIBATCH_SIZE])

        self.actor_lr = tf.placeholder(tf.float32, [])
        self.critic_lr = tf.placeholder(tf.float32, [])
        self.weight_lr = tf.placeholder(tf.float32, [])
        self.clip_range = tf.placeholder(tf.float32, [])


        new_p = train_model.dist.prob(self.action)
        self.expand_action=tf.expand_dims(self.action,axis=-1)
        old_p=tf.squeeze(tf.gather(params=self.old_ps, indices=self.expand_action, batch_dims=-1))
        ratio = new_p / tf.clip_by_value(old_p,1e-10,1.0)
        ratio =tf.reduce_mean(ratio,-1)
        # Entropy
        entropy = tf.reduce_mean(train_model.dist.entropy())

        # Critic loss
        v_pred = train_model.values  # [batch,agent]
        sch_v_pred = train_model.sch_values
        v_pred_clipped = self.old_v + tf.clip_by_value(train_model.values - self.old_v, - self.clip_range,
                                                     self.clip_range)  # [batch,agent]
        sch_v_pred_clipped = self.old_sch_v + tf.clip_by_value(train_model.sch_values - self.old_sch_v, - self.clip_range,
                                                     self.clip_range)  # [batch,agent]
        value_losses1 = tf.square(v_pred - self.returns)  # [batch,agent]
        value_losses2 = tf.square(v_pred_clipped - self.returns)  # [batch,agent]
        sch_value_losses1 = tf.square(sch_v_pred - self.sch_returns)  # [batch,agent]
        sch_value_losses2 = tf.square(sch_v_pred_clipped - self.sch_returns)  # [batch,agent]

        critic_loss = .5 * tf.reduce_mean(tf.maximum(value_losses1, value_losses2))
        sch_critic_loss = .5 * tf.reduce_mean(tf.maximum(sch_value_losses1, sch_value_losses2))
        final_critic_loss=tf.reduce_mean(critic_loss+sch_critic_loss)

        # Actor loss
        ratio=tf.squeeze(ratio)
        policy_losses = -self.advantage * ratio
        policy_losses2 = -self.advantage * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = tf.reduce_mean(tf.maximum(policy_losses, policy_losses2))
        actor_loss = policy_loss - entropy * ENTROPY_COEF

        clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

        # Training actor
        actor_params = tf.trainable_variables('ppo_model/actor_network')
        actor_trainer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
        actor_grads_and_var = actor_trainer.compute_gradients(actor_loss, actor_params)
        actor_grads, actor_var = zip(*actor_grads_and_var)
        actor_grads, actor_grad_norm = tf.clip_by_global_norm(actor_grads, MAX_GRAD_NORM)
        actor_grads_and_var = list(zip(actor_grads, actor_var))
        #  Training critic
        critic_params = tf.trainable_variables('ppo_model/critic_network')
        critic_trainer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
        critic_grads_and_var = critic_trainer.compute_gradients(final_critic_loss, critic_params)
        critic_grads, critic_var = zip(*critic_grads_and_var)
        critic_grads, critic_grad_norm = tf.clip_by_global_norm(critic_grads, MAX_GRAD_NORM)
        critic_grads_and_var = list(zip(critic_grads, critic_var))
        # Training weight net
        weight_params = tf.trainable_variables('ppo_model/schedule')
        scheduler_gradients = tf.gradients(self.train_model.sch_values, self.train_model.priority_ph)[0]
        weight_var_grads = tf.gradients(self.train_model.schedule_policy, weight_params, -scheduler_gradients)  # directly assign gradient
        weight_var_grads,weight_grad_norm = tf.clip_by_global_norm(weight_var_grads, MAX_GRAD_NORM)

        self.actor_train_op = actor_trainer.apply_gradients(actor_grads_and_var)
        self.critic_train_op = critic_trainer.apply_gradients(critic_grads_and_var)
        self.scheduler_train_op = tf.train.AdamOptimizer(self.weight_lr, epsilon=1e-5).apply_gradients(
            zip(weight_var_grads, weight_params))

        self.loss_names = ['actor_loss', 'policy_entropy', 'policy_loss','final_critic_loss', 'value_loss','sch_value_loss',
                           'clipfrac', 'actor_grad_norm',
                           'critic_grad_norm','weight_grad_norm']
        self.stats_list = [actor_loss, entropy, policy_loss,final_critic_loss, critic_loss, sch_critic_loss
                           ,clip_frac, actor_grad_norm, critic_grad_norm,weight_grad_norm]

        self.step = self.act_model.step
        self.value = self.act_model.value
        self.evalue =self.evalue_model.evalue

        initialize()

    def train(self, actor_lr, critic_lr,weight_lr, cliprange, obs, state, returns, values,sch_returns,sch_values,action,ps,
              priority, schedule_n
              ):

        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {
            self.train_model.obs: obs,
            self.train_model.state: state,
            self.advantage: advs,
            self.returns: returns,
            self.sch_returns:sch_returns,
            self.actor_lr: actor_lr,
            self.critic_lr: critic_lr,
            self.weight_lr: weight_lr,
            self.clip_range: cliprange,
            self.old_v: values,
            self.old_sch_v:sch_values,
            self.action:action,
            self.old_ps:ps,
            self.train_model.schedule_ph:schedule_n,
            self.train_model.priority_ph:priority
        }

        state = self.sess.run(self.stats_list + [self.actor_train_op, self.critic_train_op,self.scheduler_train_op], td_map)[:-3]
        return state
