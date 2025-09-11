"""MPE_Env.pyに相当"""

import importlib.util
import os.path as osp

from .DemoUser_environment import DemoUserMultiAgentEnv


def DemoUserEnv(args):
    """DemoUser_scenario.pyの内容を読み込んで、MultiAgentEnvを返す"""
    file_path = osp.join(osp.dirname(__file__), "DemoUser_scenario.py")
    module_name = "my_module_name"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario = module.Scenario()

    world = scenario.make_world(args)

    env = DemoUserMultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info,
    )
    return env
