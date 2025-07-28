import argparse


def get_config():
    """ハイパラを設定"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Reinforcement Learning (MARL)ライブラリの開発",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="シード値",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--algorithm_name",
        choices=[
            "QMIX",
            "MADDPG",
            "IPPO",
            "MAPPO",
            "HASAC",
            "ISAC",
            "HAPPO",
            "VDN",
            "MAT",
        ],
        default="MAT",
        help="アルゴリズム名の指定",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=30,
        help="エピソードの長さ",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="エージェントの数",
    )
    parser.add_argument(
        "--share_observation",
        type=bool,
        default=True,
        help="centralized_Vを計算するかどうか",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="訓練するステップ数",
    )
    return parser
