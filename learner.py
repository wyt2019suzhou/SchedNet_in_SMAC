import datetime
import os
import os.path as osp
import time

import numpy as np
import tensorflow as tf
from smac.env import StarCraft2Env

from alg_parameters import *
from env_parameters import env_args
from model import Model
from runner import Runner
from utilize import get_session, save_state


def learn(env):
    """Update model and record performance."""
    evalue_env = StarCraft2Env(**env_args)

    sess = get_session()
    # Record alg and env setting
    folderName = 'summaries/' + EXPERIMENT_NAME + datetime.datetime.now().strftime('%d-%m-%y%H%M')
    global_summary = tf.summary.FileWriter(folderName, sess.graph)
    txtPath = folderName + '/' + EXPERIMENT_NAME + '.txt'
    with open(txtPath, "w") as f:
        f.write(str(env_args))
        f.write(str(algArgs))

    model = Model(env=env)
    runner = Runner(env=env, model=model)

    num_episodes = 0
    last_test_T = -EVALUE_INTERVAL - 1
    start_time = time.perf_counter()

    for update in range(1, N_UPDATES + 1):
        frac = 1.0 - (update - 1.0) / N_UPDATES
        # Calculate learning rates
        # actorlrnow = actor_lr(frac)
        # criticlrnow = critic_lr(frac)
        # weightlrnow = weight_lr(frac)
        actorlrnow = actor_lr
        criticlrnow = critic_lr
        weightlrnow = weight_lr

        # Get experience from multiple environments
        obs, state, returns, values,sch_returns,sch_values,actions,ps,priority,schedule_n, ep_infos,performance_r = runner.run()

        mb_loss = []
        inds = np.arange(BATCH_SIZE)
        for _ in range(N_EPOCHS):
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in (obs, state, returns, values,sch_returns,sch_values, actions,ps,priority,schedule_n))
                mb_loss.append(model.train(actorlrnow, criticlrnow,weightlrnow, CLIP_RANGE, *slices))

        # Recording
        loss_vals = np.mean(mb_loss, axis=0)
        num_episodes += len(ep_infos)
        summary_episodes = tf.Summary()
        summary_step = tf.Summary()

        performance_dead_al = []
        performance_dead_enemies = []
        performance_won_rate = []
        for items in ep_infos:
            if len(items) != 3:
                continue
            else:
                performance_dead_al.append(items['dead_allies'])
                performance_dead_enemies.append(items['dead_enemies'])
                performance_won_rate.append(items['battle_won'])


        performance_dead_al = np.mean(performance_dead_al)
        performance_dead_enemies = np.mean(performance_dead_enemies)
        performance_won_rate = np.mean(performance_won_rate)

        summary_episodes.value.add(tag='Perf/Reward', simple_value=performance_r)
        summary_episodes.value.add(tag='Perf/Dead_allies', simple_value=performance_dead_al)
        summary_episodes.value.add(tag='Perf/Dead_enemies', simple_value=performance_dead_enemies)
        summary_episodes.value.add(tag='Perf/Won_rate', simple_value=performance_won_rate)
        summary_step.value.add(tag='Perf_Step/Reward_Step', simple_value=performance_r)
        summary_step.value.add(tag='Perf_Step/Dead_allies', simple_value=performance_dead_al)
        summary_step.value.add(tag='Perf_Step/Dead_enemies', simple_value=performance_dead_enemies)
        summary_step.value.add(tag='Perf_Step/Won_rate', simple_value=performance_won_rate)

        for (val, name) in zip(loss_vals, model.loss_names):
            if name == 'actor_grad_norm' or name == 'critic_grad_norm' or name == 'weight_grad_norm':
                summary_episodes.value.add(tag='grad/' + name, simple_value=val)
                summary_step.value.add(tag='grad_step/' + name, simple_value=val)
            else:
                summary_episodes.value.add(tag='loss/' + name, simple_value=val)
                summary_step.value.add(tag='loss_step/' + name, simple_value=val)
        global_summary.add_summary(summary_episodes, num_episodes)
        global_summary.add_summary(summary_step, update * BATCH_SIZE)
        global_summary.flush()

        if update % 10 == 0:
            print('update: {}, episode: {}, step: {},episode reward: {},'
                  'dead_allies: {},dead_enemies: {},won_rate: {}'.format(update, num_episodes,
                                                                         update * BATCH_SIZE, performance_r,
                                                                         performance_dead_al,
                                                                         performance_dead_enemies,
                                                                         performance_won_rate))

        if (update * BATCH_SIZE - last_test_T) / EVALUE_INTERVAL >= 1.0:
            # Evaluate model
            last_test_T = update * BATCH_SIZE
            summary_eval_step = tf.Summary()
            summary_eval_episode = tf.Summary()
            eval_reward, eval_dead_allies, eval_dead_enemies, eval_ep_len, eval_battel_won = evaluate(evalue_env, model)
            # Recording
            summary_eval_step.value.add(tag='Perf_evaluate_step/Reward', simple_value=eval_reward)
            summary_eval_step.value.add(tag='Perf_evaluate_step/Dead_allies', simple_value=eval_dead_allies)
            summary_eval_step.value.add(tag='Perf_evaluate_step/Dead_enemies', simple_value=eval_dead_enemies)
            summary_eval_step.value.add(tag='Perf_evaluate_step/Episode_len', simple_value=eval_ep_len)
            summary_eval_step.value.add(tag='Perf_evaluate_step/Won_rate', simple_value=eval_battel_won)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Reward', simple_value=eval_reward)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Dead_allies', simple_value=eval_dead_allies)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Dead_enemies', simple_value=eval_dead_enemies)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Episode_len', simple_value=eval_ep_len)
            summary_eval_episode.value.add(tag='Perf_evaluate_episode/Won_rate', simple_value=eval_battel_won)
            global_summary.add_summary(summary_eval_step, update * BATCH_SIZE)
            global_summary.add_summary(summary_eval_episode, num_episodes)
            global_summary.flush()

        if update % SAVE_INTERVAL == 0:
            tnow = time.perf_counter()
            print('consume time', tnow - start_time)
            savepath = osp.join("my_model/", '%.5i' % update)
            os.makedirs(savepath, exist_ok=True)
            savepath = osp.join(savepath, 'ppomodel')
            print('Saving to', savepath)
            save_state(savepath)

    evalue_env.close()
    return model


def evaluate(evalue_env, model):
    """Evaluate model."""
    eval_reward = []
    eval_dead_allies = []
    eval_dead_enemies = []
    eval_ep_len = []
    eval_battel_won = []
    for _ in range(EVALUE_EPISODES):
        evalue_env.reset()
        terminal = False
        episode_reward = 0
        ep_len = 0
        h_schedule_n = np.zeros((1, N_AGENTS))
        schedule_n = np.zeros((1, N_AGENTS))
        while not terminal:
            obs = evalue_env.get_obs()
            obs = np.expand_dims(obs, axis=0)
            valid_action = evalue_env.get_avail_actions()
            h_schedule_n = h_schedule_n * 0.5 + schedule_n * 0.5
            obs= np.concatenate((obs, h_schedule_n.reshape((1,N_AGENTS, 1))), axis=-1)
            schedule_n, priority = model.evalue_model.get_schedule(obs)
            actions = model.evalue(obs,schedule_n, valid_action)
            reward, terminal, info = evalue_env.step(np.squeeze(actions))
            ep_len += 1
            episode_reward += reward
            if terminal:
                eval_reward.append(episode_reward)
                eval_ep_len.append(ep_len)
                if len(info) != 3:
                    continue
                else:
                    eval_dead_allies.append(info['dead_allies'])
                    eval_dead_enemies.append(info['dead_enemies'])
                    eval_battel_won.append(info['battle_won'])
    eval_reward = np.mean(eval_reward)
    eval_dead_allies = np.mean(eval_dead_allies)
    eval_dead_enemies = np.mean(eval_dead_enemies)
    eval_ep_len = np.mean(eval_ep_len)
    eval_battel_won = np.mean(eval_battel_won)
    return eval_reward, eval_dead_allies, eval_dead_enemies, eval_ep_len, eval_battel_won
