import tensorflow as tf
from alg_parameters import *


class Net(object):
    """Build actor and critic graph."""

    def shared_dense_layer(self,observation, output_len,first_name,second_name,activation='relu'):
        """The weights of dense layer are shared."""
        all_outputs = []
        for j in range(N_AGENTS):
            first_name=first_name+str(j)
            with tf.variable_scope(first_name, reuse=tf.AUTO_REUSE):
                agent_obs = observation[:, j]
                if activation=='relu':
                    outputs = tf.layers.dense(agent_obs, output_len, name=second_name, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(0., .1),
                            bias_initializer=tf.constant_initializer(0.1),  # biases
                            use_bias=True, trainable=True, reuse=tf.AUTO_REUSE)
                if activation == 'sig':
                    outputs =tf.layers.dense(agent_obs, output_len, activation=tf.nn.sigmoid, name=second_name)
                if activation == 'none':
                    outputs =tf.layers.dense(agent_obs, output_len, name=second_name, kernel_initializer=tf.random_normal_initializer(0., .1),
                            bias_initializer=tf.constant_initializer(0.1),  # biases
                            use_bias=True, trainable=True, reuse=tf.AUTO_REUSE)
                all_outputs.append(outputs)
        all_outputs = tf.stack(all_outputs, 1)
        return all_outputs

    def encoder_network(self,observation):
        hidden=self.shared_dense_layer(observation,h_num,"encoder","encoder")
        ret=self.shared_dense_layer(hidden,capacity,"encoder","encoder_out")
        return ret

    def decode_concat_network(self,encoder, schedule,name):
        reshaped_messages=[]
        for i in range(N_AGENTS):
            name=name+str(i)
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                masked_msg = tf.boolean_mask(encoder,tf.cast(schedule, tf.bool),axis=0)
                reshaped_messages.append(masked_msg)
        return tf.stack(reshaped_messages, 1)

    def comm_encoded_obs(self,observation, aggr_out):
        c_input = tf.concat([observation, aggr_out], axis=-1)
        hidden_1=self.shared_dense_layer(c_input,h_num,"comm","sender_1")
        hidden_3 = self.shared_dense_layer(hidden_1, h_num, "comm", "sender_3")
        hidden_4 = self.shared_dense_layer(hidden_3, N_ACTIONS, "comm", "sender_4",activation='none')
        return hidden_4

    def generate_actor_network(self, observation,schedule):
        with tf.variable_scope('actor_network', reuse=tf.AUTO_REUSE):
            encoder=self.encoder_network(observation)
            aggr_out =self.decode_concat_network(encoder, schedule,'aggr')
            agent_actor_logits = self.comm_encoded_obs(observation, aggr_out)
        return agent_actor_logits

    def generate_wg(self,obs):
        with tf.variable_scope('schedule', reuse=tf.AUTO_REUSE):
            hidden_1=self.shared_dense_layer(obs,h1_scheduler,"schedule","weight_1")
            schedule =self.shared_dense_layer(hidden_1, 1, "schedule", "weight_3",activation='sig')
        return schedule

    def generate_critic_network(self,state, priority):
        with tf.variable_scope('critic_network', reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(state, h1_critic, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     use_bias=True, name='dense_c1')

            hidden_2 = tf.layers.dense(hidden, h2_critic, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(0., .1),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       use_bias=True,  name='dense_c2')

            q_values = tf.layers.dense(hidden_2, 1,
                                       kernel_initializer=tf.random_normal_initializer(0., .1),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name='dense_c4', use_bias=False)
            h2_sch = tf.concat([hidden_2, priority], axis=1)

            sch_hidden_3 = tf.layers.dense(h2_sch, h3_critic, activation=tf.nn.relu,
                                           kernel_initializer=tf.random_normal_initializer(0., .1),
                                           bias_initializer=tf.constant_initializer(0.1),
                                           use_bias=True, name='dense_c3_sch')

            sch_q_values = tf.layers.dense(sch_hidden_3, 1,
                                           kernel_initializer=tf.random_normal_initializer(0., .1),
                                           bias_initializer=tf.constant_initializer(0.1),
                                           name='dense_c4_sch', use_bias=False)

        return q_values,sch_q_values
