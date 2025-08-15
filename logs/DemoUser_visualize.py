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
    dir_name = f"logs/episode_{episode}"
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
    goals = None
    for i in range(4):
        if goals is None:
            goals = obs_list[0][0][4 + i * 4 + 2 : 4 + i * 4 + 4]
        else:
            goals = np.concatenate(
                (goals, obs_list[0][0][4 + i * 4 + 2 : 4 + i * 4 + 4])
            )

    for t in range(len(obs_list) + 1):
        fig, ax = plt.subplots(figsize=(4, 4))
        if t == 0:
            # 初期状態
            coors = np.array([0, 0, 0, 5, 5, 5, 5, 0])
        else:
            coors = None
            for i in range(4):
                if coors is None:
                    coors = obs_list[t - 1][0][4 + i * 4 : 4 + i * 4 + 2]
                else:
                    coors = np.concatenate(
                        (coors, obs_list[t - 1][0][4 + i * 4 : 4 + i * 4 + 2])
                    )
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

        # フレーム保存
        plt.savefig(f"logs/episode_{episode}/{t}.png")
        plt.close()
    # GIF作成
    with imageio.get_writer(
        f"logs/episode_{episode}/DemoUser.gif", duration=5.0
    ) as writer:
        for t in range(len(obs_list) + 1):
            image = imageio.imread(f"logs/episode_{episode}/{t}.png")
            writer.append_data(image)

    print("✅ GIF保存完了: DemoUser.gif")
