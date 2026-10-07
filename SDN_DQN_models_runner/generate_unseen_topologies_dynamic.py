"""Generate unseen 25-node / 50-link topologies for Dynamic-only evaluation.

This script preserves:
- 25 nodes
- 50 links
- source node 0 / destination node 24
- degree sequence through double-edge swaps
- link-type counts and state-vector slot order

It generates 10 topologies for each rewiring ratio:
20%, 40%, and 60% (30 topologies total).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from topology import NUM_NODES, SOURCE, DESTINATION, EDGES, EDGE_TYPES


OUTPUT_DIR = Path("topologies_dynamic")
TOPOLOGY_COUNT_PER_RATIO = 10
REWIRE_RATIOS = (0.20, 0.40, 0.60)
BASE_SEED = 20260
MAX_ATTEMPTS = 1000

Edge = Tuple[int, int]


def normalize_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def edge_type_for(edge: Edge) -> str:
    e = normalize_edge(*edge)
    return EDGE_TYPES.get(
        e,
        EDGE_TYPES.get((e[1], e[0]), "normal"),
    )


def graph_is_valid(graph: nx.Graph) -> bool:
    """Check whether a generated topology is usable for evaluation."""
    if graph.number_of_nodes() != NUM_NODES:
        return False

    if graph.number_of_edges() != len(EDGES):
        return False

    if not nx.is_connected(graph):
        return False

    if SOURCE not in graph or DESTINATION not in graph:
        return False

    if graph.degree(SOURCE) == 0 or graph.degree(DESTINATION) == 0:
        return False

    try:
        shortest_hops = nx.shortest_path_length(
            graph,
            SOURCE,
            DESTINATION,
        )
    except nx.NetworkXNoPath:
        return False

    # Reject trivial or excessively long source-destination structures.
    return 4 <= shortest_hops <= 12


def generate_rewired_edges(ratio: float, seed: int) -> List[Edge]:
    """Rewire edges while preserving node degrees and total edge count."""
    original = [normalize_edge(*edge) for edge in EDGES]
    original_set = set(original)

    target_changed = max(
        2,
        int(round(len(original) * ratio)),
    )

    for attempt in range(MAX_ATTEMPTS):
        graph = nx.Graph()
        graph.add_nodes_from(range(NUM_NODES))
        graph.add_edges_from(original)

        attempt_seed = seed + attempt * 1009

        # A double-edge swap replaces two edges.
        nswap = max(
            1,
            int(round(target_changed / 2)),
        )
        max_tries = max(
            200,
            nswap * 200,
        )

        try:
            nx.double_edge_swap(
                graph,
                nswap=nswap,
                max_tries=max_tries,
                seed=attempt_seed,
            )
        except nx.NetworkXAlgorithmError:
            continue

        if not graph_is_valid(graph):
            continue

        new_set = {
            normalize_edge(*edge)
            for edge in graph.edges()
        }

        changed_count = len(original_set - new_set)

        # Require at least 75% of the requested structural change.
        if changed_count < max(
            2,
            int(target_changed * 0.75),
        ):
            continue

        removed_indices = [
            index
            for index, edge in enumerate(original)
            if edge not in new_set
        ]
        added_edges = sorted(new_set - original_set)

        if len(removed_indices) != len(added_edges):
            continue

        # Preserve state-vector slot count and ordering.
        ordered_edges = list(original)
        for index, new_edge in zip(
            removed_indices,
            added_edges,
        ):
            ordered_edges[index] = new_edge

        if len(set(ordered_edges)) != len(ordered_edges):
            continue

        return ordered_edges

    raise RuntimeError(
        "Failed to generate a valid topology "
        f"for ratio={ratio:.2f}, seed={seed}."
    )


def save_topology(
    edges: List[Edge],
    ratio: float,
    index: int,
    seed: int,
) -> Path:
    """Save one unseen topology as JSON."""
    edge_records = []

    # The edge occupying each state slot inherits the original slot's link type.
    for slot, (original_edge, new_edge) in enumerate(
        zip(EDGES, edges)
    ):
        edge_records.append(
            {
                "slot": slot,
                "u": int(new_edge[0]),
                "v": int(new_edge[1]),
                "type": edge_type_for(original_edge),
            }
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(NUM_NODES))
    graph.add_edges_from(edges)

    name = (
        f"unseen_dynamic_r{int(ratio * 100):02d}_"
        f"{index:02d}"
    )

    payload: Dict[str, object] = {
        "name": name,
        "num_nodes": NUM_NODES,
        "source": SOURCE,
        "destination": DESTINATION,
        "num_edges": len(edges),
        "rewire_ratio_requested": ratio,
        "seed": seed,
        "shortest_path_hops": nx.shortest_path_length(
            graph,
            SOURCE,
            DESTINATION,
        ),
        "average_degree": (
            sum(dict(graph.degree()).values()) / NUM_NODES
        ),
        "edges": edge_records,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    print("Generating Dynamic-only unseen topologies")
    print(
        f"Nodes={NUM_NODES}, "
        f"Edges={len(EDGES)}, "
        f"Source={SOURCE}, "
        f"Destination={DESTINATION}"
    )
    print(
        "Ratios:",
        REWIRE_RATIOS,
        "| Topologies per ratio:",
        TOPOLOGY_COUNT_PER_RATIO,
    )

    generated: List[Path] = []
    counter = 0

    for ratio in REWIRE_RATIOS:
        for index in range(
            1,
            TOPOLOGY_COUNT_PER_RATIO + 1,
        ):
            seed = BASE_SEED + counter

            edges = generate_rewired_edges(
                ratio=ratio,
                seed=seed,
            )

            path = save_topology(
                edges=edges,
                ratio=ratio,
                index=index,
                seed=seed,
            )

            generated.append(path)
            counter += 1
            print("Saved:", path)

    print(
        f"\nDone. Generated {len(generated)} topologies "
        f"in '{OUTPUT_DIR}'."
    )


if __name__ == "__main__":
    main()
