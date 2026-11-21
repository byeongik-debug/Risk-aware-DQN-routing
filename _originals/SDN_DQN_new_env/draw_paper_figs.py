import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 그림 2: Episode Reward Curve (window=100) ──────────────────────
data = np.load("training_data.npz")
episodes   = data["episodes"]
ep_rewards = data["ep_rewards"]

w = 100
smoothed  = np.convolve(ep_rewards, np.ones(w) / w, mode='valid')
smooth_ep = np.arange(w, len(ep_rewards) + 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(episodes, ep_rewards, color='#F4A97B', linewidth=0.5, alpha=0.4, label='Episode Reward (raw)')
ax.plot(smooth_ep, smoothed, color='#C05621', linewidth=2.0, label=f'Moving Avg (n={w})')
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Episode Reward', fontsize=12)
ax.tick_params(labelsize=12)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig2_reward.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("저장: fig2_reward.png")

# ── 그림 3: Q-value Convergence (window=200) ──────────────────────
data2    = np.load("q_data_dynamic.npz")
episodes2 = data2["episodes"]
q_values  = data2["q_values"]

w2 = 200
smoothed2  = np.convolve(q_values, np.ones(w2) / w2, mode='valid')
smooth_ep2 = episodes2[w2 - 1:]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(episodes2, q_values, color='#A8C4E0', linewidth=0.6, alpha=0.5, label='Max Q-value (raw)')
ax.plot(smooth_ep2, smoothed2, color='#2B6CB0', linewidth=2.0, label=f'Moving Average (n={w2})')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Max Q-value at Start State', fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig3_qvalue.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("저장: fig3_qvalue.png")
