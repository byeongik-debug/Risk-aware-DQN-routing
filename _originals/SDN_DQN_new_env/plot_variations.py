# plot_variations.py
# 다양한 설정으로 그래프를 한 번에 생성해서 비교할 수 있게 저장

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_PATH  = "training_data.npz"
OUT_DIR    = "graph_variations"
os.makedirs(OUT_DIR, exist_ok=True)

data            = np.load(DATA_PATH)
episodes        = data["episodes"]
q_values        = data["q_values"]
ep_rewards      = data["ep_rewards"]
ep_losses       = data["ep_losses"]
ep_success      = data["ep_success"]
ep_hops         = data["ep_hops"]
ep_avg_risk     = data["ep_avg_risk"]
ep_risky_ratio  = data["ep_risky_ratio"]
ep_epsilon      = data["ep_epsilon"]
all_ep          = np.arange(1, len(ep_rewards) + 1)


def moving_avg(arr, window):
    if len(arr) < window:
        return None, None
    smoothed = np.convolve(arr, np.ones(window) / window, mode='valid')
    ep       = np.arange(window, len(arr) + 1)
    return ep, smoothed


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"저장: {path}")


# ── 1. Q값 / 이동평균 윈도우 비교 ─────────────────────────────────
for w in [50, 100, 200, 500]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, q_values, color='#A8C4E0', linewidth=0.5, alpha=0.4, label='Q-value (raw)')
    ep_s, s = moving_avg(q_values, w)
    if s is not None:
        ax.plot(ep_s, s, color='#2B6CB0', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Max Q-value')
    ax.set_title(f'Q-value Convergence  |  window={w}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'q_window{w}.png')

# ── 2. Q값 / raw 없이 이동평균만 ──────────────────────────────────
for w in [100, 200]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep_s, s = moving_avg(q_values, w)
    if s is not None:
        ax.plot(ep_s, s, color='#2B6CB0', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Max Q-value')
    ax.set_title(f'Q-value Convergence (smoothed only)  |  window={w}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'q_smooth_only_window{w}.png')

# ── 3. 보상 / 이동평균 윈도우 비교 ───────────────────────────────
for w in [50, 100, 200, 500]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, ep_rewards, color='#F4A97B', linewidth=0.5, alpha=0.4, label='Episode Reward (raw)')
    ep_s, s = moving_avg(ep_rewards, w)
    if s is not None:
        ax.plot(ep_s, s, color='#C05621', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode', fontsize=11); ax.set_ylabel('Episode Reward', fontsize=11)
    ax.legend(loc='lower right', fontsize=11); ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'reward_window{w}.png')

# ── 4. 보상 / raw 없이 이동평균만 ────────────────────────────────
for w in [100, 200]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep_s, s = moving_avg(ep_rewards, w)
    if s is not None:
        ax.plot(ep_s, s, color='#C05621', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Episode Reward')
    ax.set_title(f'Episode Reward Curve (smoothed only)  |  window={w}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'reward_smooth_only_window{w}.png')

# ── 5. Q값 + 보상 나란히 (subplot) ───────────────────────────────
for w in [100, 200]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(episodes, q_values, color='#A8C4E0', linewidth=0.5, alpha=0.4)
    ep_s, s = moving_avg(q_values, w)
    if s is not None:
        ax1.plot(ep_s, s, color='#2B6CB0', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Max Q-value')
    ax1.set_title('Q-value Convergence')
    ax1.legend(loc='lower right'); ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.plot(episodes, ep_rewards, color='#F4A97B', linewidth=0.5, alpha=0.4)
    ep_s, s = moving_avg(ep_rewards, w)
    if s is not None:
        ax2.plot(ep_s, s, color='#C05621', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Reward')
    ax2.set_title('Episode Reward Curve')
    ax2.legend(loc='lower right'); ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.suptitle(f'Training Convergence (Dynamic Scenario)  |  window={w}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, f'combined_window{w}.png')

# ── 6. TD Loss / 이동평균 윈도우 비교 ────────────────────────────
for w in [50, 100, 200, 500]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_ep, ep_losses, color='#B794F4', linewidth=0.5, alpha=0.4, label='TD Loss (raw)')
    ep_s, s = moving_avg(ep_losses, w)
    if s is not None:
        ax.plot(ep_s, s, color='#6B46C1', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('TD Loss')
    ax.set_title(f'TD Loss (Training Loss)  |  window={w}')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'loss_window{w}.png')

# ── 7. TD Loss smooth only ────────────────────────────────────────
for w in [100, 200]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep_s, s = moving_avg(ep_losses, w)
    if s is not None:
        ax.plot(ep_s, s, color='#6B46C1', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('TD Loss')
    ax.set_title(f'TD Loss (smoothed only)  |  window={w}')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'loss_smooth_only_window{w}.png')

# ── 8. 성공률 / 이동평균 윈도우 비교 ─────────────────────────────
for w in [50, 100, 200, 500]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep_s, s = moving_avg(ep_success, w)
    if s is not None:
        ax.plot(ep_s, s * 100, color='#276749', linewidth=2.0, label=f'Success Rate % (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'Success Rate during Training  |  window={w}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'success_window{w}.png')

# ── 9. Risky Link Ratio ───────────────────────────────────────────
for w in [50, 100, 200, 500]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_ep, ep_risky_ratio * 100, color='#FBB6CE', linewidth=0.5, alpha=0.4, label='Risky Link Ratio (raw)')
    ep_s, s = moving_avg(ep_risky_ratio * 100, w)
    if s is not None:
        ax.plot(ep_s, s, color='#C53030', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Risky Link Ratio (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'Risky Link Ratio during Training  |  window={w}')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'risky_ratio_window{w}.png')

# ── 10. Average Link Risk ─────────────────────────────────────────
for w in [100, 200]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_ep, ep_avg_risk, color='#FBD38D', linewidth=0.5, alpha=0.4, label='Avg Risk (raw)')
    ep_s, s = moving_avg(ep_avg_risk, w)
    if s is not None:
        ax.plot(ep_s, s, color='#C05621', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Average Link Risk')
    ax.set_title(f'Average Link Risk during Training  |  window={w}')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'avg_risk_window{w}.png')

# ── 11. Hop Count ─────────────────────────────────────────────────
for w in [100, 200]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ep_s, s = moving_avg(ep_hops, w)
    if s is not None:
        ax.plot(ep_s, s, color='#2C7A7B', linewidth=2.0, label=f'Moving Avg (n={w})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Hop Count')
    ax.set_title(f'Hop Count during Training  |  window={w}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, f'hops_window{w}.png')

# ── 12. Epsilon decay ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(all_ep, ep_epsilon, color='#4A5568', linewidth=1.5, label='Epsilon')
ax.set_xlabel('Episode'); ax.set_ylabel('Epsilon')
ax.set_title('Epsilon Decay during Training')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
save(fig, 'epsilon_decay.png')

print(f"\n총 {len(os.listdir(OUT_DIR))}개 그래프 저장 완료 → {OUT_DIR}/")
