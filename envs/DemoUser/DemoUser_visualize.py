import webbrowser
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np


def visualizer(
    episode,
    obs_list,
    reward_list,
    action_list,
):
    agent_colors = ["red", "orange", "purple", "green"]
    grid_size = 6
    dir_name = f"./envs/DemoUser/logs/episode_{episode:04}"
    os.makedirs(dir_name, exist_ok=True)
    forbiddens = [
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
    goals = []
    for i in range(4):
        goals = np.concatenate((goals, obs_list[0][i][6:8]))
    for t in range(len(obs_list) + 1):
        fig, ax = plt.subplots(figsize=(4, 4))
        coors = []
        if t == 0:
            # 初期状態
            coors = np.array([0, 0, 0, 5, 5, 5, 5, 0])
        else:
            for i in range(4):
                coors = np.concatenate((coors, obs_list[t - 1][i][4:6]))
            # coors = obs_list[t - 1][0][4:]
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_title(f"Episode {episode}, Step {t}")
        ax.grid(True)

        # ゴールを描画
        for i in range(4):
            ax.plot(
                goals[i * 2],
                goals[i * 2 + 1],
                "s",
                color=agent_colors[i],
                markersize=20,
            )

        # 黒ブロックを描画
        for black_block in forbiddens:
            ax.plot(
                black_block[0],
                black_block[1],
                "s",
                color="black",
                markersize=20,
            )

        # 描画：エージェント
        for i in range(4):
            x, y = coors[i * 2], coors[i * 2 + 1]
            ax.plot(x, y, "o", color=agent_colors[i], markersize=20)
            # 報酬の値を表示（t>0のときのみ）
            # 矢印・バツの描画

            _visualize_action(ax, x, y, action_list[t][i][0])
            _visualize_reward(
                ax,
                x,
                y,
                reward_list[t][i][0],
                color="blue" if reward_list[t][i][0] >= 0 else "red",
            )

        # フレーム保存
        plt.savefig(dir_name + f"/{t}.png")
        plt.close()
    # GIF作成
    with imageio.get_writer(dir_name + f"/DemoUser.gif", duration=5.0) as writer:
        for t in range(len(obs_list) + 1):
            image = imageio.imread(dir_name + f"/{t}.png")
            writer.append_data(image)

    print("✅ GIF保存完了: DemoUser.gif")


def _visualize_action(ax, x, y, action):
    """(x, y)の位置に、actionに応じた矢印や記号をテキストで表示する"""

    # アクションに対応する記号を定義
    action_symbols = {
        0: "←",  # 左
        1: "→",  # 右
        2: "↓",  # 下
        3: "↑",  # 上
        4: "×",  # バツ
    }

    symbol = action_symbols.get(action)

    if symbol:
        ax.text(
            x,
            y,
            symbol,
            ha="center",  # 水平方向の中央揃え
            va="center",  # 垂直方向の中央揃え
            fontsize=20,  # フォントサイズは適宜調整してください
            color="black",
        )


def _visualize_reward(ax, x, y, reward, color):
    """Draw the reward value above the agent at (x, y) on ax."""
    ax.text(
        x,
        y + 0.4,
        f"{reward:.2f}",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
