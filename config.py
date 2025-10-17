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
        default="VDN",  # VDNを動かすために変更
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
        "--share_observation",
        type=bool,
        default=True,
        help="centralized_Vを計算するかどうか",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=160e6,  # QMIXを動かすために10e6から160e6に変更
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

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_true",
        default=True,  # QMIXを動かすためにTrueに変更
        help="Whether to use the same policy for all agents",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_ReLU",
        action="store_false",
        default=True,
        help="Whether to use ReLU",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.01,
        help="The gain # of last action layer",
    )

    # recurrent parameters
    parser.add_argument(
        "--recurrent_N",
        type=int,
        default=3,
        help="The number of recurrent layers.",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=True,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
    )

    # ppo parameters
    parser.add_argument(
        "--ppo_epoch",
        type=int,
        default=15,
        help="number of ppo epochs (default: 15)",
    )
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=10.0,
        help=" coefficience of huber loss.",
    )
    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=16000,  # 50から16000に変更
        help="time duration between contiunous twice log printing.",
    )

    # add for transformer
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--num_embd", type=int, default=64)
    parser.add_argument("--num_head", type=int, default=1)

    # add for QMIX
    parser.add_argument(
        "--qmix_buffer_size",
        type=int,
        default=10000,
        help="QMIX用: リプレイバッファの最大エピソード数",
    )
    parser.add_argument(
        "--qmix_batch_size",
        type=int,
        default=512,
        help="QMIX用: 学習時にサンプリングするエピソード数",
    )
    parser.add_argument(
        "--qmix_target_update_interval",
        type=int,
        default=1000,
        help="QMIX用: ターゲットネットワークの更新間隔（学習回数）",
    )
    parser.add_argument(
        "--qmix_epsilon_start",
        type=float,
        default=1.0,
        help="QMIX用: ε-greedy探索の初期値",
    )
    parser.add_argument(
        "--qmix_epsilon_final",
        type=float,
        default=0.05,
        help="QMIX用: ε-greedy探索の最終値",
    )
    parser.add_argument(
        "--qmix_epsilon_anneal_time",
        type=int,
        default=130e6,
        help="QMIX用: εを減衰させるステップ数",
    )
    parser.add_argument(
        "--qmix_mixer_embed_dim",
        type=int,
        default=32,
        help="QMIX用: Mixerネットワークの埋め込み次元",
    )
    parser.add_argument(
        "--qmix_mixer_hidden_size",
        type=int,
        default=32,
        help="QMIX用: Mixerネットワークの隠れ層サイズ",
    )
    parser.add_argument(
        "--qmix_rnn_hidden_dim",
        type=int,
        default=256,
        help="RNNの隠れ状態の次元",
    )
    parser.add_argument(
        "--qmix_hypernet_layers",
        type=int,
        default=2,
        help="QMIX用: ハイパーネットワークの層数",
    )
    parser.add_argument(
        "--qmix_hypernet_embed_dim",
        type=int,
        default=64,
        help="QMIX用: ハイパーネットワークの埋め込み次元（層数が2の場合に使用）",
    )
    parser.add_argument(
        "--qmix_gamma",
        type=float,
        default=0.99,
        help="QMIX用: 割引率",
    )
    return parser


# 試行１: uv run python main.py --log_interval=1000 --lr=5e-8 --num_env_steps=20000000 --num_rollout_threads=48
# 結果: うまくいかず　イプシロンが減衰しきった後の学習が進まない　おそらくepisode_lengthが短すぎてゴールまでたどり着けていない
# uv run python main.py --log_interval=100 --num_env_steps=100000000 --num_rollout_threads=32 --qmix_epsilon_anneal_time=75000000 --episode_length=100
# 結果: うまくいかず　急にLossが２０から４００に増大した。おそらく、target networkの更新が早すぎるのでは？
# target networkの更新間隔を100から2000に変更
# uv run python main.py --log_interval=1000 --num_env_steps=100000000 --num_rollout_threads=32 --qmix_epsilon_anneal_time=75000000 --episode_length=100 --lr=5e-8
