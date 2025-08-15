class EntityState(object):
    def __init__(self):
        # 座標
        self.coor = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()


class GoalState(EntityState):
    def __init__(self):
        super(GoalState, self).__init__()


class Action(object):
    def __init__(self):
        # 上下左右の動き
        self.move = None


class Agent:
    def __init__(self):
        self.state = AgentState()
        self.goal = GoalState()
        self.action = Action()


# multi-agent world
class World:
    def __init__(self):
        self.agents = []
        self.goals = []
        self.forbiddens = [
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
        self.world_step = 0

    def step(
        self,
    ):
        self.world_step += 1
        occupied = []
        for i, agent in enumerate(self.agents):
            assumed_coor = agent.state.coor + agent.action.move

            if (
                0 <= assumed_coor[0] < 6
                and 0 <= assumed_coor[1] < 6
                and tuple(assumed_coor) not in map(tuple, self.forbiddens)
            ):  # 枠からははみ出ないし、かつ禁止エリアでもない
                flag = False
                for j in range(i + 1, len(self.agents)):
                    if (
                        assumed_coor == self.agents[j].state.coor
                    ).all():  # まだ動いていないエージェントにぶつかるかの判定
                        flag = True
                        break
                if flag:  # まだ動いていないエージェントにぶつかる
                    occupied.append(agent.state.coor)
                else:
                    if tuple(assumed_coor) in map(
                        tuple, occupied
                    ):  # 既に動いたエージェントにぶつかる
                        occupied.append(agent.state.coor)
                    else:
                        agent.state.coor = assumed_coor  # 動けるので座標を更新
                        occupied.append(assumed_coor)
            else:
                occupied.append(agent.state.coor)  # 枠からはみ出た場合はそのまま
