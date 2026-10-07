# test_ospf.py
# OSPF Bandwidth-based Routing Animation

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from network_env import NetworkEnv
from topology import EDGES, SOURCE, DESTINATION

SAVE_DIR = "results"
SAVE_PATH = os.path.join(SAVE_DIR, "ospf_routing.gif")

def get_fixed_positions():
    return {
        0: (0, 5), 1: (2, 5), 2: (4, 5), 3: (6, 5),
        4: (0, 4), 5: (2, 4), 6: (4, 4), 7: (6, 4),
        8: (-1, 3), 9: (1, 3), 10: (3, 3), 11: (5, 3), 12: (7, 3),
        13: (-1, 2), 14: (1, 2), 15: (3, 2), 16: (5, 2), 17: (7, 2), 18: (9, 2),
        19: (0, 1), 20: (2, 1), 21: (4, 1), 22: (6, 1),
        23: (3, 0), 24: (5, 0),
    }

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    env = NetworkEnv(use_risk=False, scenario="dynamic")
    env.reset()
    
    for _ in range(10):
        env.update_dynamic_qos()

    # 대역폭 코스트 기준으로 탐색하도록 연동
    ospf_path = env.ospf_path(cost_type="bandwidth") 
    metrics = env.calculate_path_metrics(ospf_path)
    
    print("OSPF (Bandwidth) Selected Path:", ospf_path)
    print("Metrics:", metrics)

    G = nx.Graph()
    G.add_edges_from(EDGES)
    pos = get_fixed_positions()

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(step):
        ax.clear()
        
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color="lightgray", ax=ax)
        
        congestion_set = {env.normalize_edge(*x) for x in [(0, 4), (4, 8), (8, 14), (14, 20), (20, 23), (23, 24)]}
        unstable_set = {env.normalize_edge(*x) for x in [(0, 1), (1, 5), (5, 10), (10, 15), (15, 21), (21, 22), (22, 24)]}
        
        for u, v in EDGES:
            e = env.normalize_edge(u, v)
            if e in congestion_set:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.5, edge_color="purple", ax=ax)
            elif e in unstable_set:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.5, edge_color="orange", ax=ax)

        node_colors = []
        for n in range(25):
            if n == SOURCE: node_colors.append("lime")
            elif n == DESTINATION: node_colors.append("red")
            elif n in ospf_path[:step+1]: node_colors.append("yellow")
            else: node_colors.append("skyblue")

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

        if step > 0:
            path_edges = list(zip(ospf_path[:step], ospf_path[1:step+1]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color="red", ax=ax)

        curr_node = ospf_path[step]
        ax.plot(pos[curr_node][0], pos[curr_node][1], marker="o", color="darkred", markersize=12, zorder=12)

        info_text = (
            f"Algorithm: OSPF\\n"
            f"Cost metric: Max Bandwidth\\n"
            f"Path: {ospf_path}\\n"
            f"Hop: {metrics['hop_count']}\\n"
            f"Delay: {metrics['total_delay']:.2f} ms\\n"
            f"Loss: {metrics['total_loss']:.4f}\\n"
            f"Risk: {metrics['total_risk']:.4f}\\n"
            f"Unstable links: {metrics['unstable_count']}\\n"
            f"Congested links: {metrics['congested_count']}"
        )
        ax.text(8.2, 4.5, info_text, fontsize=9, bbox=dict(facecolor="white", edgecolor="black", alpha=0.9))

        legend_text = (
            "Legend\\n"
            "Red Line: OSPF path\\n"
            "Orange Link: Unstable\\n"
            "Purple Link: Congested\\n"
            "Lime Node: Source (0)\\n"
            "Red Node: Destination (24)"
        )
        ax.text(-1.8, 4.5, legend_text, fontsize=9, bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
        
        ax.set_title(f"OSPF Bandwidth-based Routing Simulation (Step {step})", fontsize=14, fontweight="bold")
        ax.axis("off")

    anim = animation.FuncAnimation(fig, update, frames=len(ospf_path), interval=800, repeat=True)
    anim.save(SAVE_PATH, writer="imageio")
    plt.close()
    print(f"OSPF 애니메이션 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    main()