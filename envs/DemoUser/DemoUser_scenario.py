import sys

import numpy as np

sys.path.append("../../")
from envs.DemoUser.DemoUser_core import Agent, World


# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def info(self, agent, world):
        return {}


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        world.num_agents = args.num_agents

        # add agents
        world.agents = [Agent() for _ in range(world.num_agents)]
        self.reset_world(world=world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            if i == 0:  # 赤エージェント
                agent.state.coor = [0, 0]
                agent.goal.coor = [5, 5]
            elif i == 1:  # オレンジエージェント
                agent.state.coor = [0, 5]
                agent.goal.coor = [5, 0]
            elif i == 2:  # 紫エージェント
                agent.state.coor = [5, 5]
                agent.goal.coor = [0, 0]
            elif i == 3:
                agent.state.coor = [5, 0]
                agent.goal.coor = [0, 5]

    def reward(self, agent, world):
        pass

    def observation(self, agent, world):
        # すべてのエージェントの座標をconcatしたもの
        obs = []
        for agent_i in world.agents:
            obs.append(agent_i.state.coor)
        return np.concatenate(obs)
