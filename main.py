from config import get_config
import torch

from envs.env_wrappers import SubprocVecEnv
from runner.main_runner import UserEnvRunner as Runner


def make_train_envs(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "DemoUser":
                from envs.DemoUser.DemoUser_createEnv import DemoUserEnv as UserEnv
            env = UserEnv(args)
            env.seed(args.seed + rank)
            return env

        return init_env

    return SubprocVecEnv(
        [get_env_fn(i) for i in range(args.num_rollout_threads)],
    )


def main(args):
    # 訓練環境を初期化する
    envs = make_train_envs(args)

    config = {
        "args": args,
        "envs": envs,
    }

    # runnerを作る
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    parser = get_config()

    if torch.cuda.is_available():
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
    else:
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
        )

    args = parser.parse_args()
    main(args)
