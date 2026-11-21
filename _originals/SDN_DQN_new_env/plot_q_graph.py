# plot_q_graph.py
# q_data_dynamic.npz 데이터를 불러와 Q값 그래프를 다시 그리는 스크립트
# 학습 없이 그래프만 조정할 때 사용

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH    = "q_data_dynamic.npz"
Q_GRAPH_PATH = "q_convergence_dynamic.png"
R_GRAPH_PATH = "reward_convergence_dynamic.png"

# ── 여기서 그래프 설정 변경 ──────────────────────────
Q_SMOOTH_WINDOW = 200   # Q값 이동 평균 윈도우
R_SMOOTH_WINDOW = 200   # 보상 이동 평균 윈도우
SHOW_RAW        = True  # 원시 데이터 표시 여부
# ────────────────────────────────────────────────────


def plot_q(data):
    episodes = data["episodes"]
    q_values = data["q_values"]

    fig, ax = plt.subplots(figsize=(8, 4))
    if SHOW_RAW:
        ax.plot(episodes, q_values, color='#A8C4E0', linewidth=0.6,
                alpha=0.5, label='Max Q-value (raw)')
    if len(q_values) >= Q_SMOOTH_WINDOW:
        smoothed  = np.convolve(q_values, np.ones(Q_SMOOTH_WINDOW) / Q_SMOOTH_WINDOW, mode='valid')
        smooth_ep = episodes[Q_SMOOTH_WINDOW - 1:]
        ax.plot(smooth_ep, smoothed, color='#2B6CB0', linewidth=2.0,
                label=f'Moving Average (n={Q_SMOOTH_WINDOW})')

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Max Q-value at Start State', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(Q_GRAPH_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Q값 그래프 저장: {Q_GRAPH_PATH}")


def plot_reward(data):
    ep_rewards = data["ep_rewards"]
    episodes   = np.arange(1, len(ep_rewards) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    if SHOW_RAW:
        ax.plot(episodes, ep_rewards, color='#F4A97B', linewidth=0.6,
                alpha=0.4, label='Episode Reward')
    if len(ep_rewards) >= R_SMOOTH_WINDOW:
        smoothed  = np.convolve(ep_rewards, np.ones(R_SMOOTH_WINDOW) / R_SMOOTH_WINDOW, mode='valid')
        smooth_ep = episodes[R_SMOOTH_WINDOW - 1:]
        ax.plot(smooth_ep, smoothed, color='#C05621', linewidth=2.0,
                label=f'Moving Average (n={R_SMOOTH_WINDOW})')

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(R_GRAPH_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"보상 그래프 저장: {R_GRAPH_PATH}")


if __name__ == "__main__":
    data = np.load(DATA_PATH)
    plot_q(data)
    plot_reward(data)