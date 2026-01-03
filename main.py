import importlib

import torch

from config import get_config
from envs.env_wrappers import SubprocVecEnv


def make_train_envs(args):
    def get_env_fn(rank):
        def init_env():
            createEnvClass = importlib.import_module(
                f"envs.{args.user_name}.{args.user_name}_createEnv"
            )
            UserEnv = getattr(createEnvClass, f"{args.user_name}Env")
            env = UserEnv(args)
            env.seed(args.seed + rank)
            return env

        return init_env

    return SubprocVecEnv(
        [get_env_fn(i) for i in range(args.num_rollout_threads)],
    )


def main(args):
    # 訓練環境を初期化する
    envs: SubprocVecEnv = make_train_envs(args)

    if args.algorithm_name == "IPPO":
        args.use_centralized_V = False
    config = {
        "args": args,
        "envs": envs,
    }

    # runnerを作る
    if args.share_policy:
        from runner.shared.main_runner import UserEnvRunner as Runner
    else:
        from runner.separated.main_runner import UserEnvRunner as Runner

    # HASACを動かすためのコード
    if args.algorithm_name == "HASAC":
        from runner.separated.offpolicy_main_runner import OffPolicyMainRunner as Runner

    # QMIXを動かすためのコード
    if args.algorithm_name == "QMIX" or args.algorithm_name == "VDN":
        from runner.shared.offpolicy_main_runner import OffPolicyMainRunner as Runner

    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    parser = get_config()

    if torch.cuda.is_available():
        print("We will use GPU to train the model.")
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
    else:
        print("We will use CPU to train the model.")
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
        )

    args = parser.parse_args()
    main(args)
