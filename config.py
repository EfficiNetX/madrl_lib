import argparse
import os

import yaml


def get_config():
    pre_perser = argparse.ArgumentParser(add_help=False)
    pre_perser.add_argument(
        "--algorithm_name",
        default="MAT",
        choices=[
            "QMIX",
            "MADDPG",
            "IPPO",
            "RMAPPO",
            "HASAC",
            "ISAC",
            "HAPPO",
            "VDN",
            "MAT",
            "MAT_DEC",
        ],
        help="Yamlファイルを選択するためのアルゴリズム名",
    )
    pre_args, _ = pre_perser.parse_known_args()
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
        "--num_rollout_threads",
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
            "RMAPPO",
            "HASAC",
            "ISAC",
            "HAPPO",
            "VDN",
            "MAT",
            "MAT_DEC",
        ],
        default="MAT",
        help="アルゴリズム名の指定",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=50,
        help="エピソードの長さ",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="エージェントの数",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="訓練するステップ数",
    )
    # env parameters
    parser.add_argument(
        "--user_name",
        type=str,
        default="DemoUser",
        choices=["DemoUser", "LogisticsUser"],
        help="環境名の指定",
    )
    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="time duration between contiunous twice log printing.",
    )
    # アルゴリズムに応じてyamlファイルからハイパラを読み込む
    config_dir = "config"
    defaults = {}
    type_map = {"int": int, "float": float, "str": str, "bool": bool}
    config_path = f"{config_dir}/{pre_args.algorithm_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} is not found.")

    print(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"{config_path} is empty.")

    if "arguments" not in data:
        raise KeyError(f"'arguments' key is not found in {config_path}.")
    for name, params in data["arguments"].items():
        arg_name = f"--{name}"
        arg_type_str = params.get("type", "str")
        arg_type = type_map.get(arg_type_str, str)
        arg_help = params.get("help", "")
        default_val = params.get("default")

        is_readonly = params.get("readonly", False)
        defaults[name] = default_val
        if is_readonly:
            continue

        if arg_type == bool:
            action = "store_true" if not default_val else "store_false"
            parser.add_argument(
                arg_name,
                action=action,
                default=default_val,
                help=arg_help,
            )
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=default_val,
                help=arg_help,
            )

    parser.set_defaults(**defaults)

    return parser


if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    print("--- Configurations ---")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
