import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語の文字化け対策

# データの読み込み
# ファイルにヘッダー（列名）がないため header=None を指定
try:
    df = pd.read_csv("loss_log.txt", header=None, names=["学習回数", "loss", "報酬"])

    # --- グラフの描画 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 1. Lossのグラフ
    ax1.plot(df["学習回数"], df["loss"], color="tab:blue", label="Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("学習回数ごとのLossの推移")
    ax1.grid(True)
    ax1.legend()

    # 2. 報酬のグラフ
    ax2.plot(df["学習回数"], df["報酬"], color="tab:orange", label="報酬")
    ax2.set_xlabel("学習回数")
    ax2.set_ylabel("報酬")
    ax2.set_title("学習回数ごとの報酬の推移")
    ax2.grid(True)
    ax2.legend()

    # レイアウトの調整とグラフの表示
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("エラー: loss_log.txt が見つかりません。")
    print("提示されたデータを 'loss_log.txt' という名前で保存してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
