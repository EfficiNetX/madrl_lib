from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

import numpy as np


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """SubprocVecEnvのための抽象クラス"""

    def __init__(
        self,
        num_envs,
        observation_space,
        share_observation_space,
        action_space,
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
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

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()


def worker(
    remote,
    parent_remote,
    env_fn_wrapper,
):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done = env.step(data)
            if "bool" in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done))
        elif cmd == "reset":
            ob = env.reset()
            remote.send((ob))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (
                    env.observation_space,
                    env.share_observation_space,
                    env.action_space,
                )
            )
        elif cmd == "get_avail_actions":
            avail_actions = env.get_avail_actions()
            remote.send(avail_actions)
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(
        self,
        env_fns,
    ):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                ),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))

        (
            observation_space,
            share_observation_space,
            action_space,
        ) = self.remotes[0].recv()

        ShareVecEnv.__init__(
            self,
            len(env_fns),
            observation_space,
            share_observation_space,
            action_space,
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        (obs, rews, dones) = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def get_avail_actions(self):
        for remote in self.remotes:
            remote.send(("get_avail_actions", None))
        avail_actions = [remote.recv() for remote in self.remotes]
        return np.stack(avail_actions)
