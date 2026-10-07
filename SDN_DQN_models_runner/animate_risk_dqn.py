# animate_risk_dqn.py
# Risk-Only DQN Routing Animation

import os
import imageio
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from topology import (
    NUM_NODES,
    SOURCE,
    DESTINATION,
    EDGES,
    EDGE_TYPES
)

from network_env import NetworkEnv


# ============================================================
# 설정
# ============================================================

SCENARIO = "dynamic"
MODEL_PATH = "risk_only_model_final.pth"
RESULT_DIR = "results"
GIF_NAME = "risk_only_dqn_routing.gif"


# ============================================================
# Q-Network
# 학습 코드와 반드시 동일해야 함
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Environment & Model Load
# ============================================================

env = NetworkEnv(
    use_risk=True,
    scenario=SCENARIO
)

model = QNetwork(
    env.state_size,
    env.action_size
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# Graph
# ============================================================

G = nx.Graph()
G.add_nodes_from(range(NUM_NODES))
G.add_edges_from(EDGES)

pos = nx.spring_layout(
    G,
    seed=42
)


# ============================================================
# Routing
# ============================================================

state = env.reset()
done = False
path = [SOURCE]

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)[0]

    valid_actions = env.get_valid_actions()

    if len(valid_actions) == 0:
        break

    masked_q = torch.full_like(
        q_values,
        -1e9
    )

    for action in valid_actions:
        masked_q[action] = q_values[action]

    action = int(torch.argmax(masked_q).item())

    next_state, reward, done, info = env.step(action)

    path.append(action)
    state = next_state

    if len(path) > 30:
        break


# ============================================================
# Create Result Folder
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

frames = []


# ============================================================
# Draw Frames
# ============================================================

for step in range(len(path)):
    plt.figure(figsize=(12, 8))

    normal_edges = [
        e for e, t in EDGE_TYPES.items()
        if t == "normal"
    ]

    backbone_edges = [
        e for e, t in EDGE_TYPES.items()
        if t == "backbone"
    ]

    backup_edges = [
        e for e, t in EDGE_TYPES.items()
        if t == "backup"
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        width=2,
        edge_color="gray"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=backbone_edges,
        width=4,
        edge_color="blue"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=backup_edges,
        width=2,
        style="dashed",
        edge_color="green"
    )

    node_colors = []

    for node in G.nodes():
        if node == SOURCE:
            node_colors.append("lime")
        elif node == DESTINATION:
            node_colors.append("red")
        elif node in path[:step + 1]:
            node_colors.append("orange")
        else:
            node_colors.append("skyblue")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=700,
        node_color=node_colors
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold"
    )

    if step > 0:
        path_edges = list(
            zip(
                path[:step],
                path[1:step + 1]
            )
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            width=5,
            edge_color="red"
        )

    plt.title(
        f"Risk-Only DQN Routing\n"
        f"Scenario: {SCENARIO} | Step {step}",
        fontsize=18
    )

    filename = os.path.join(
        RESULT_DIR,
        f"risk_only_frame_{step}.png"
    )

    plt.savefig(filename)
    plt.close()

    frames.append(
        imageio.imread(filename)
    )


# ============================================================
# Save GIF
# ============================================================

gif_path = os.path.join(
    RESULT_DIR,
    GIF_NAME
)

imageio.mimsave(
    gif_path,
    frames,
    duration=0.8
)

print("\nGIF 저장 완료:")
print(gif_path)

print("\nRouting Path:")
print(path)