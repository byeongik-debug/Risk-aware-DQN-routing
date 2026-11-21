"""NetworkEnv variant that scores paths with a window-independent risk yardstick.

The window sweep compares policies trained with different temporal windows, but the
reported ``total_risk`` is produced by the very formula the window parameterises.  A
deque of maxlen 1 has zero standard deviation, so at window=1 both instability factors
(delay jitter, weight 0.15; loss jitter, weight 0.10) are identically zero and the risk
score is mechanically smaller.  Comparing those numbers measures the ruler, not the route.

This environment keeps a second pair of delay/loss histories pinned to ``ref_window``
(default 5) and exposes the resulting ``risk_ref`` per link and ``total_risk_ref`` per
path.  The agent's observation and reward still use ``risk`` from its own training
window, because a policy must be evaluated on the state distribution it was trained on.

The overrides read ``self.qos`` and append to the reference histories only.  They draw
no random numbers, so an episode unfolds exactly as it would under NetworkEnv.
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SDN_DQN_new_env"))

from network_env import (  # noqa: E402
    CONGESTION_LINKS,
    RISK_WINDOW,
    UNSTABLE_LINKS,
    NetworkEnv,
)
from risk_calculator import calculate_link_risk  # noqa: E402
from topology import EDGES  # noqa: E402

REFERENCE_WINDOW = 5


class FixedYardstickEnv(NetworkEnv):
    """NetworkEnv plus a risk score measured on a fixed reference window."""

    def __init__(self, use_risk=True, scenario="normal", risk_window=RISK_WINDOW,
                 ref_window=REFERENCE_WINDOW):
        self.ref_window = int(ref_window)
        self.ref_loss_history = {}
        self.ref_delay_history = {}
        super().__init__(use_risk=use_risk, scenario=scenario, risk_window=risk_window)

    def _ensure_ref_histories(self):
        """Create the reference deques lazily; __init__ may trigger init_qos early."""
        if len(self.ref_loss_history) == len(EDGES):
            return
        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            self.ref_loss_history[e] = deque(maxlen=self.ref_window)
            self.ref_delay_history[e] = deque(maxlen=self.ref_window)

    def _reference_risk_type(self, e, link_type, danger_set):
        """Reproduce the risk_type label update_dynamic_qos assigns to this link."""
        if self.scenario == "congestion" and e in {self.normalize_edge(*x) for x in CONGESTION_LINKS}:
            return "congested"
        if self.scenario == "unstable" and e in {self.normalize_edge(*x) for x in UNSTABLE_LINKS}:
            return "unstable"
        if self.scenario == "dynamic" and e in danger_set:
            return "unstable_congested"
        return link_type

    def init_qos(self):
        """Seed the reference histories alongside the training ones.

        Mirrors NetworkEnv.init_qos, which computes the initial risk with no scenario
        scaling applied -- the 0.60/0.55 factors live in update_dynamic_qos only.
        """
        super().init_qos()
        self._ensure_ref_histories()
        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]
            self.ref_loss_history[e].clear()
            self.ref_delay_history[e].clear()
            self.ref_loss_history[e].append(q["loss"])
            self.ref_delay_history[e].append(q["delay"])
            q["risk_ref"] = calculate_link_risk(
                utilization=q["utilization"],
                is_congested=q["is_congested"],
                delay_history=self.ref_delay_history[e],
                loss_history=self.ref_loss_history[e],
                link_type=q["type"],
            )

    def update_dynamic_qos(self):
        """Advance the environment, then score every link on the reference window too."""
        super().update_dynamic_qos()
        self._ensure_ref_histories()

        danger_set, support_set = (set(), set())
        if self.scenario == "dynamic":
            danger_set, support_set = self.get_dynamic_phase_sets()

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]
            link_type = q["type"]

            self.ref_loss_history[e].append(q["loss"])
            self.ref_delay_history[e].append(q["delay"])

            risk_ref = calculate_link_risk(
                utilization=q["utilization"],
                is_congested=q["is_congested"],
                delay_history=self.ref_delay_history[e],
                loss_history=self.ref_loss_history[e],
                link_type=self._reference_risk_type(e, link_type, danger_set),
            )

            if self.scenario in ["congestion", "unstable"] and link_type == "backup":
                risk_ref = float(np.clip(risk_ref * 0.60, 0.0, 1.0))

            if self.scenario == "dynamic" and e in support_set:
                risk_ref = float(np.clip(risk_ref * 0.55, 0.0, 1.0))

            q["risk_ref"] = risk_ref

    def calculate_path_metrics(self, path):
        """NetworkEnv metrics plus total_risk_ref on the fixed reference window."""
        metrics = super().calculate_path_metrics(path)

        total_risk_ref = 0.0
        hop_count = max(len(path) - 1, 1)
        for u, v in zip(path[:-1], path[1:]):
            e = self.normalize_edge(u, v)
            if e not in self.qos:
                continue
            total_risk_ref += self.qos[e].get("risk_ref", 0.0)

        metrics["total_risk_ref"] = total_risk_ref / hop_count
        return metrics
