import json


class EntityState(object):
    def __init__(self):
        # ノード番号
        self.id = None
        self.name = None
        self.x = None
        self.z = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()


class BuildingState(EntityState):
    def __init__(self):
        super(BuildingState, self).__init__()


class Action(object):
    def __init__(self):
        # どのBuildingに向かって動くか
        self.move = None


class Agent:
    def __init__(self):
        self.state = AgentState()
        self.action = Action()


class Building:
    def __init__(self):
        self.state = BuildingState


# multi-agent world
class World:
    def __init__(self):
        self.agents = []
        self.buildings = []
        self.intermediate_nodes = []
        self.world_step = 0
