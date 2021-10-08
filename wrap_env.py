import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
from smac.env import StarCraft2Env

from env_parameters import env_args


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):  # enable to pickle
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_envs, state_shap, obs_shape, n_actions, n_agents, episode_limit):
        self.num_envs = num_envs
        self.num_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shap
        self.cent_state_shape = self.obs_shape + self.state_shape
        self.n_actions = n_actions
        self.episode_limit = episode_limit

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()


def make_train_env(nenv):
    def get_env_fn(rank):
        def init_env():
            env = StarCraft2Env(**env_args)
            return env
        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(nenv)])


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    """

    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   # use in default process
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_env_info', None))
        info_dic = self.remotes[0].recv()
        VecEnv.__init__(self, self.nenvs, info_dic['state_shape'], info_dic['obs_shape'],
                        info_dic['n_actions'], info_dic['n_agents'], info_dic['episode_limit'])

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nenvs)  # split all env action to one env action
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        reward, dones, infos = zip(*results)
        return np.stack(reward), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_state(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_state', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = np.stack(obs)
        return np.expand_dims(obs, axis=1)

    def get_avail_actions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_avail_actions', None))
        actions = [remote.recv() for remote in self.remotes]
        return np.stack(actions)

    def get_obs(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_obs', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        reward, done, info = env.step(action[0])
        if done:
            env.reset()
        return (reward, done, info)

    parent_remote.close()
    envs = env_fn_wrappers.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(step_env(envs, data))
            elif cmd == 'reset':
                remote.send(envs.reset())
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_env_info':
                remote.send(envs.get_env_info())
            elif cmd == 'get_state':
                remote.send(envs.get_state())
            elif cmd == 'get_obs':
                remote.send(envs.get_obs())
            elif cmd == 'get_avail_actions':
                remote.send(envs.get_avail_actions())
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
