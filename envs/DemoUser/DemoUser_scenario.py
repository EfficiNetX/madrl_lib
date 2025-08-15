import numpy as np

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
        occupied = [
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [1, 3],
            [2, 3],
            [3, 3],
            [4, 3],
            [1, 5],
            [2, 5],
            [3, 5],
            [4, 5],
        ]
        for i, agent in enumerate(world.agents):
            while True:
                pos = np.random.randint(0, 6, size=2).tolist()
                if pos not in occupied:
                    agent.goal.coor = pos
                    occupied.append(pos)
                    break

            if i == 0:  # 赤エージェント
                agent.state.coor = [0, 0]
                # agent.goal.coor = [5, 5]
            elif i == 1:  # オレンジエージェント
                agent.state.coor = [0, 5]
                # agent.goal.coor = [5, 0]
            elif i == 2:  # 紫エージェント
                agent.state.coor = [5, 5]
                # agent.goal.coor = [0, 0]
            elif i == 3:
                agent.state.coor = [5, 0]
                # agent.goal.coor = [0, 5]

    def reward(self, agent, world):
        # 報酬は各エージェントのゴールまでの距離（を負にしたもの）
        rew = 0
        for agent_i in world.agents:
            # rew -= np.linalg.norm(
            #     np.array(agent_i.state.coor) - np.array(agent_i.goal.coor)
            # )

            # rewardがスパースの場合
            if np.array_equal(agent_i.state.coor, agent_i.goal.coor):
                rew += 1
        return rew

    def observation(self, agent, world):
        # すべてのエージェントの座標をconcatしたもの
        obs = []
        for agent_i in world.agents:
            obs.append(agent_i.state.coor)
            obs.append(agent_i.goal.coor)
        return np.concatenate(obs)
