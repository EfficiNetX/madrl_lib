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
