"""C-1(노드 이동성) 축소형 측정 — 에피소드 중간 링크 단절/복구.

왜 새로 재는가
--------------
R3-4의 `unseen_dynamic_r20/40/60`은 **토폴로지마다 정적**이다. 에피소드가 시작되면
간선 집합이 끝까지 고정이므로, 그 실험은 "다른 토폴로지로 일반화하는가"를 재고
"움직이는 토폴로지에서 동작하는가"는 **한 번도 재지 않았다**. 이동성의 최소 형태는
흐름이 진행되는 **도중에** 링크가 사라지고 돌아오는 것이다. 이 파일이 그걸 잰다.

환경이 이동성을 어디까지 표현할 수 있는가 (코드 근거)
-----------------------------------------------------
- `network_env.py` `get_state()`는 `self.graph`가 아니라 **모듈 전역 `EDGES`를
  순회**한다. 상태의 risk 블록은 고정 길이 50이므로 **끊긴 링크를 비울 자리가 없다.**
  따라서 "링크 소멸"을 그대로는 관측에 넣을 수 없다. 이것이 환경을 다시 쓰지 않고
  이동성을 넣을 수 없다는 주장의 정확한 코드 근거다.
- 반면 `get_valid_actions()`는 `self.graph.neighbors()`를 **호출 시점에** 읽고,
  `shortest_distance_to_destination()`도 `self.graph`를 읽으며, `step()`은 죽은
  링크를 고르면 `reason="invalid"`로 즉시 종료한다(보상 -10). 즉 **행동 마스킹과
  보상은 이동성을 따라오고 관측만 못 따라온다.**

그래서 축소형을 이렇게 정했다
-----------------------------
끊긴 링크는 (1) `self.graph`에서 제거해 마스크가 막게 하고, (2) `qos[e]["risk"]`를
1.0으로 올려 고정 인코딩 안에서 "최악의 링크"로 보이게 한다. 동결 모델은 이런 패턴을
학습한 적이 없으므로 이건 **설계상 분포 밖 시험**이다 — C-1이 묻는 것이 바로 그것이다.

교란을 막기 위해 지킨 것
------------------------
1. `update_dynamic_qos()`를 **추가로 호출하지 않는다.** `network_env.py:228-229`에서
   지연·손실이 누산되므로, 호출 횟수가 달라지면 이동성과 무관한 차이가 생긴다.
   risk 표시는 그 함수를 덮어써서 super() 뒤에 다시 찍는 방식으로만 한다.
2. 고장은 `random.Random((seed, ep, t))`로 뽑는다. 전역 RNG를 쓰면 두 방법의 경로
   길이가 달라 소비량이 어긋나 **서로 다른 고장 수열**을 보게 된다. 스텝 인덱스로
   키를 잡으면 같은 t에 같은 고장을 본다.
3. SOURCE→DESTINATION을 분리하는 고장은 **거부**하고 횟수를 센다. 거부하지 않으면
   도달 불가를 정책 품질로 오독한다.
4. OSPF 비교군은 `self.graph`를 읽는 재계획기를 새로 썼다. `reeval_ospf_replan.py`의
   `ospf_plan_excluding`은 `RLC.ADJ`(정적 인접 테이블)를 읽고 `blocked`가 노드 단위라
   **링크 고장을 담을 수 없다** — 그대로 쓰면 OSPF가 없는 링크로 라우팅한다.
5. 양쪽 모두 **재방문 배제** 디코딩을 쓴다(`FINDINGS_topology_generalization.md` §4의
   2-주기 결함). 안 그러면 이동성이 아니라 그 결함을 재게 된다.
6. `--churn 0`이면 평범한 `NetworkEnv`를 경로까지 그대로 재현해야 한다. `verify()`가
   그걸 검사하고, 어긋나면 측정 전에 멈춘다.

읽는 방식
---------
도달률이 1차 지표다. 품질 지표는 실현값 규약(`walk_realized`)만 쓴다 — 끊긴 링크의
risk가 1.0으로 찍혀 있으므로 최종 상태로 경로를 재채점하는 규약(`walk_final`)은
이동성 실험에서 뜻이 틀어진다.

실행:  python -I -X utf8 -B reeval_mobility_proxy.py --episodes 120
출력:  review_results/mobility_proxy.csv
"""

import argparse
import csv
import math
import os
import random
import sys

import networkx as nx
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
import reeval_ospf_replan as ROR  # noqa: E402
import reeval_scoring_convention as RSC  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import (fmt_p, holm_bonferroni, mean_std,  # noqa: E402
                          min_detectable_effect, t_critical,
                          t_two_sided_p, welch)
from topology import DESTINATION, EDGES, NUM_NODES, SOURCE  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "mobility_proxy.csv")

# 평가 래퍼의 디코딩 규칙. 기본값 "all"이 본 측정에 쓴 규칙이므로 기존 CSV가
# 그대로 재현된다. 6-1절에서 이 규칙이 실패를 **제조한다**는 것을 쟀으므로
# "prev"(직전 노드만 배제 = 2-주기만 막는다)로 다시 돌려 비교한다.
DECODE_RULE = "all"

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
WARMUP = RLC.WARMUP

# 스텝당 고장이 하나 발생할 확률. 0.0은 회귀 검사용 기준선이다.
CHURN_PS = (0.0, 0.20, 0.50)
# "link" = 링크 한 개 단절, "node" = 노드 하나가 자기 인접 링크 전체를 떨어뜨림,
# "node_protected" = 같되 흐름이 서 있는 노드는 떠나지 않음.
# 리뷰어 C-1이 묻는 것은 노드 이동성이므로 "node"가 본 답이고, "link"는 약한
# 대조군, "node_protected"는 실패가 전부 자기 노드 이탈인지 가리는 대조군이다.
CHURN_UNITS = ("link", "node", "node_protected")
DOWN_STEPS = 3          # 내려간 링크가 돌아오기까지의 스텝 수
DOWN_RISK = 1.0         # 고정 인코딩 안에서 고장을 표현하는 값
PROTECT_CURRENT = False  # 반증 시험에서만 True (진단 전용, 본 측정 아님)

METRICS = ("hop", "delay", "published", "path_sum", "compounded", "cong_live")
# `stuck_decode`는 재방문 배제 규칙이 만든 실패다 — 환경이 아니라 내 측정의
# 인공물이므로 따로 세어 보고한다.
REASONS = ("invalid", "step_limit", "stuck_isolated", "stuck_nopath",
           "stuck_decode")

realized_metrics = RSC.realized_metrics
seed_mean = RLC.seed_mean
ospf_cost = ROR.ospf_cost


# --------------------------------------------------------------- 환경


class MobileEnv(NetworkEnv):
    """에피소드 중간에 링크가 내려가고 돌아오는 환경.

    `churn_p`가 0이면 부모 클래스와 완전히 같게 동작해야 한다(`verify()`가 검사).
    """

    def __init__(self, churn_p=0.0, churn_key=(0, 0), churn_unit="link",
                 mark_risk=True, mark_value=None, **kwargs):
        self.churn_p = churn_p
        self.churn_key = churn_key
        self.churn_unit = churn_unit
        self.mark_risk = mark_risk
        # 표시 **크기**를 바꿀 수 있게 둔다. 그래프에서 간선을 빼는 것은 그대로고
        # 상태 벡터에 쓰는 값만 달라진다 — 고장 자체와 고장 신호를 분리한다.
        self.mark_value = DOWN_RISK if mark_value is None else mark_value
        self.down = {}          # normalize_edge -> 남은 스텝 수
        self.clock = 0
        self.rejected = 0       # 연결을 끊어서 거부된 고장 횟수
        self.down_steps_seen = 0
        super().__init__(**kwargs)

    # -- 고정 인코딩 안에서 고장을 보이게 하는 유일한 지점 -------------
    def _mark_down(self):
        if not self.mark_risk:
            return      # 절제 실험: 고장은 내되 상태 벡터에는 알리지 않는다
        for e in self.down:
            q = self.qos.get(e)
            if q is not None:
                q["risk"] = self.mark_value

    def update_dynamic_qos(self):
        super().update_dynamic_qos()
        self._mark_down()       # QoS를 다시 돌리지 않고 표시만 덧씌운다

    def reset(self):
        # 이전 에피소드에서 뺀 간선을 되돌린다. add_edges_from은 멱등이다.
        self.graph.add_edges_from(EDGES)
        self.down.clear()
        self.clock = 0
        self.rejected = 0
        self.down_steps_seen = 0
        state = super().reset()
        if self.churn_p > 0.0:
            self._advance_churn()
            self._mark_down()
            state = self.get_state()
        return state

    # -- 고장 수열은 (seed, episode, step)만의 함수다 ------------------
    def _advance_churn(self):
        self.clock += 1

        for e in list(self.down):
            self.down[e] -= 1
            if self.down[e] <= 0:
                del self.down[e]
                self.graph.add_edge(*e)

        rng = random.Random("%d-%d-%d" % (self.churn_key[0],
                                          self.churn_key[1], self.clock))
        if rng.random() >= self.churn_p:
            self.down_steps_seen += len(self.down)
            return

        if self.churn_unit in ("node", "node_protected"):
            # 움직이는 노드는 자기 인접 링크를 한꺼번에 떨어뜨린다.
            # 후보에서 양 끝단만 뺀다 — **현재 노드는 빼지 않는다.** 스텝 t에
            # 두 방법이 서 있는 위치가 다르므로 현재 노드를 제외하면 뽑히는
            # 노드가 방법마다 달라져 "같은 고장 수열" 전제가 깨진다. 서 있는
            # 노드가 고립되면 valid_actions가 비어 stuck으로 끝나는데, 그것은
            # 정책이 스스로를 취약한 위치에 두었는지를 재는 측정값이다.
            cands = [n for n in range(NUM_NODES)
                     if n not in (SOURCE, DESTINATION)]
            if self.churn_unit == "node_protected" or PROTECT_CURRENT:
                # 대조군. 서 있는 노드를 후보에서 빼면 두 방법이 서로 다른
                # 고장을 보게 되지만(위치가 다르다), "실패가 전부 자기 노드
                # 이탈인가"를 가리는 데는 그게 바로 필요한 조작이다.
                cands = [n for n in cands if n != self.current_node]
            node = cands[rng.randrange(len(cands))]
            edges = [self.normalize_edge(node, nb)
                     for nb in self.graph.neighbors(node)]
            edges = [e for e in edges if e not in self.down]
        else:
            u, v = EDGES[rng.randrange(len(EDGES))]
            e = self.normalize_edge(u, v)
            edges = [e] if (e not in self.down
                            and self.graph.has_edge(*e)) else []

        if not edges:
            self.down_steps_seen += len(self.down)
            return

        self.graph.remove_edges_from(edges)
        if nx.has_path(self.graph, SOURCE, DESTINATION):
            for e in edges:
                self.down[e] = DOWN_STEPS
        else:
            self.graph.add_edges_from(edges)   # 도달 불가 고장은 거부한다
            self.rejected += 1
        self.down_steps_seen += len(self.down)

    def step(self, action):
        out = super().step(action)
        if self.churn_p > 0.0:
            self._advance_churn()
            self._mark_down()
        return out


# --------------------------------------------------------------- 비교군


def plan_live(env, blocked, cost_type):
    """현재 노드에서 출발하는 최단경로. **살아 있는 `env.graph`만** 쓴다.

    `ROR.ospf_plan_excluding`과 비용 함수는 같지만 인접 관계를 정적 테이블이 아니라
    그래프에서 읽는다 — 그래야 OSPF도 링크 고장을 본다.
    """
    import heapq

    source = env.current_node
    dist = {source: 0.0}
    prev = {source: None}
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for nb in env.graph.neighbors(node):
            if nb == source or nb in blocked:
                continue
            q = env.qos.get(env.normalize_edge(node, nb))
            if q is None:
                continue
            new_cost = cost + ospf_cost(q, cost_type)
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    if DESTINATION not in prev:
        return []
    path, curr = [], DESTINATION
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return path


# --------------------------------------------------------------- 보행


def diagnose_stuck(env):
    """막힌 원인을 셋으로 가른다 — 내 디코딩 규칙이 만든 막힘을 분리하려고.

    재방문 배제는 정적 환경의 2-주기 결함을 막으려 넣은 규칙이지만, 이동성에서는
    **되돌아가기가 정당한 복구 동작**이다. 그 규칙이 실패를 제조했다면 그건
    환경이 아니라 내 측정의 인공물이므로 분리해서 보고해야 한다.
    """
    node = env.current_node
    if env.graph.degree(node) == 0:
        return "stuck_isolated"        # 이웃이 아예 없다 — 누구도 못 고친다
    if not nx.has_path(env.graph, node, DESTINATION):
        return "stuck_nopath"          # 재방문을 허용해도 목적지가 끊겼다
    return "stuck_decode"              # 되돌아갈 수 있었는데 규칙이 막았다


def walk(env, next_node_of):
    """전진하며 실현값을 모은다. 간선은 step 전에, 값은 step 후에 읽는다."""
    samples = []
    done = False
    reason = "ok"
    while not done:
        nxt = next_node_of(env)
        if nxt is None:
            reason = diagnose_stuck(env)
            break
        e = env.normalize_edge(env.current_node, nxt)
        _, _, done, info = env.step(nxt)
        q = env.qos.get(e)
        if q is not None:
            samples.append((q["delay"], q["loss"],
                            bool(q.get("is_congested", False))))
        if isinstance(info, dict) and info.get("reason") == "invalid":
            reason = "invalid"
            break
        if len(env.visited) >= MAX_STEPS:
            reason = "step_limit"
            break
    return samples, reason


def make_env(scenario, churn_p, churn_key, churn_unit="link",
             mark_risk=True, mark_value=None):
    env = MobileEnv(churn_p=churn_p, churn_key=churn_key,
                    churn_unit=churn_unit, mark_risk=mark_risk,
                    mark_value=mark_value,
                    use_risk=True, scenario=scenario,
                    risk_window=RLC.RISK_WINDOW)
    env.reset()
    for _ in range(WARMUP):
        env.update_dynamic_qos()
    return env


def run_dqn(scenario, seed, episodes, churn_p, model, churn_unit="link"):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    rows, reasons, rejected, downs, per_ep = [], [], 0, [], []

    for ep in range(episodes):
        env = make_env(scenario, churn_p, (seed, ep), churn_unit)

        def next_node_of(e):
            # 재방문 배제 디코딩 (FINDINGS_topology_generalization.md §4)
            bad = blocked_by(e, DECODE_RULE)
            valid = [a for a in e.get_valid_actions() if a not in bad]
            if not valid:
                return None
            state = e.get_state()
            with torch.no_grad():
                q = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            return int(torch.argmax(masked).item())

        samples, reason = walk(env, next_node_of)
        reasons.append(reason)
        rejected += env.rejected
        downs.append(env.down_steps_seen / max(env.clock, 1))
        if env.current_node == DESTINATION:
            m = realized_metrics(samples)
            rows.append(m)
            per_ep.append(m)
        else:
            per_ep.append(None)

    return _summary(rows, reasons, rejected, downs, episodes), per_ep


def run_ospf(scenario, seed, episodes, churn_p, cost_type, churn_unit="link"):
    np.random.seed(seed)
    random.seed(seed)
    rows, reasons, rejected, downs, per_ep = [], [], 0, [], []

    for ep in range(episodes):
        env = make_env(scenario, churn_p, (seed, ep), churn_unit)

        def next_node_of(e):
            # 매 홉 재계획 + 재방문 배제. 비교군을 최대한 유리하게 잡는다.
            path = plan_live(e, blocked_by(e, DECODE_RULE), cost_type)
            if len(path) < 2:
                return None
            return path[1]

        samples, reason = walk(env, next_node_of)
        reasons.append(reason)
        rejected += env.rejected
        downs.append(env.down_steps_seen / max(env.clock, 1))
        if env.current_node == DESTINATION:
            m = realized_metrics(samples)
            rows.append(m)
            per_ep.append(m)
        else:
            per_ep.append(None)

    return _summary(rows, reasons, rejected, downs, episodes), per_ep


def _summary(rows, reasons, rejected, downs, episodes):
    out = dict(seed_mean(rows)) if rows else {m: float("nan") for m in METRICS}
    out["success"] = len(rows) / float(episodes) * 100.0
    for r in REASONS:
        out[r] = reasons.count(r) / float(episodes) * 100.0
    out["rejected_per_ep"] = rejected / float(episodes)
    out["mean_down_links"] = float(np.mean(downs)) if downs else 0.0
    return out


def paired_metrics(per_ep, mask):
    """**세 방법이 모두 도달한 에피소드만으로** 품질 지표를 다시 잰다.

    도달률이 다른 방법끼리 품질 지표를 비교하면 생존자 편향이 생긴다 — 어려운
    에피소드에서 탈락한 쪽이 더 좋아 보인다. 세 방법이 같은 `(seed, ep)` 고장
    수열을 쓰므로 에피소드 인덱스로 교집합을 잡을 수 있다.

    기준을 제안 방법 자신으로 두면 안 된다. 그러면 제안 방법 행은 자기가 도달한
    에피소드 **전부**를 쓰고 비교군 행만 교집합으로 줄어들어, 편향을 없애려 넣은
    장치가 거꾸로 비대칭을 만든다(도달률 85% 구간이면 최대 15%의 에피소드가 서로
    다른 집합이 된다). 그래서 마스크를 셀마다 한 번 만들어 세 행에 **같은 집합**을
    쓴다.
    """
    both = [m for m, keep in zip(per_ep, mask)
            if keep and m is not None]
    out = {"paired_n": len(both)}
    if both:
        mean = seed_mean(both)
        out.update({"paired_" + k: mean.get(k) for k in METRICS})
    else:
        out.update({"paired_" + k: float("nan") for k in METRICS})
    return out


VARIANTS = (
    ("risk_dqn", None),
    ("ospf_bw_replan", "bandwidth"),
    ("ospf_delay_replan", "delay"),
)


# --------------------------------------------------------------- 검증


def verify_visible(scenario="unstable"):
    """고장이 **실제로 보이는지** 확인한다 (양성 대조).

    `churn_p=0` 게이트는 "고장이 없을 때 환경이 안 바뀐다"만 증명한다. 끊긴 간선이
    그래프·QoS 표·상태 벡터의 **올바른 자리**·행동 마스크·플래너에 전부 닿는지는
    별개 문제이고, 안 닿으면 이 실험의 "차이 없음"은 이동성이 아니라 내 계측
    결함을 재는 것이 된다. 그래서 측정 전에 다섯 경로를 하나씩 확인한다.
    """
    print("양성 대조 — 고장이 그래프·QoS·상태·마스크·플래너에 닿는가")
    bad = []
    for unit in ("link", "node"):
        before = len(bad)
        random.seed(3)
        np.random.seed(3)
        env = make_env(scenario, 1.0, (3, 0), unit)
        env.reset()
        if not env.down:
            bad.append("%s: reset 뒤 끊긴 간선이 없다" % unit)
            continue
        st = env.get_state()
        index = {}
        for i, (u, v) in enumerate(EDGES):
            index[env.normalize_edge(u, v)] = i
        for e in env.down:
            if env.graph.has_edge(*e):
                bad.append("%s: 그래프에 %s가 남아 있다" % (unit, (e,)))
            if env.qos[e]["risk"] != DOWN_RISK:
                bad.append("%s: qos risk가 안 찍혔다 %s" % (unit, (e,)))
            if st[index[e]] != DOWN_RISK:
                bad.append("%s: 상태 %d번 자리가 안 찍혔다" % (unit, index[e]))
        if not set(env.get_valid_actions()) <= set(
                env.graph.neighbors(env.current_node)):
            bad.append("%s: 마스크가 죽은 이웃을 허용한다" % unit)
        path = plan_live(env, set(env.visited), "delay")
        for k in range(len(path) - 1):
            if not env.graph.has_edge(path[k], path[k + 1]):
                bad.append("%s: 플래너가 죽은 간선을 썼다" % unit)
        was = list(env.down)
        env.churn_p = 0.0
        for _ in range(DOWN_STEPS):
            env._advance_churn()
            env._mark_down()
        if env.down or not all(env.graph.has_edge(*e) for e in was):
            bad.append("%s: %d스텝 뒤 복구되지 않았다" % (unit, DOWN_STEPS))
        print("  %-5s 끊긴 간선 %d개, 상태 자리 %s → %s"
              % (unit, len(was), sorted(index[e] for e in was),
                 "통과" if len(bad) == before else "실패"))
    if bad:
        for m in bad:
            print("    " + m)
        raise SystemExit("양성 대조 실패 — 고장이 계측에 닿지 않는다")
    print("  통과: 고장이 다섯 경로 전부에 닿고 복구된다")
    print()


def dqn_picker(model, device):
    """동결 모델의 결정적 argmax 정책. 유효 행동 밖은 -1e9로 눌러 마스크한다."""
    import torch

    def next_node_of(e):
        valid = [a for a in e.get_valid_actions() if a not in e.visited]
        if not valid:
            return None
        st = e.get_state()
        with torch.no_grad():
            q = model(torch.FloatTensor(st).unsqueeze(0).to(device))[0]
        masked = torch.full_like(q, -1e9)
        for a in valid:
            masked[a] = q[a]
        return int(torch.argmax(masked).item())

    return next_node_of


def measure_path_variety(episodes, churn_p=0.0):
    """고장 없이, 정책은 몇 가지 길을 걷는가.

    7-3절에서 `congestion`과 `dynamic`은 risk 분포가 거의 같은데(95분위 0.733 대
    0.750) **분포 안** 표시에 대한 경로 변화가 4.0% 대 26.3%로 갈렸다. 분포로는
    설명이 안 된다. 가설: "`congestion`은 혼잡 집합이 고정이라 정책이 하나의
    우회를 외웠고, `dynamic`은 `reset()`이 에피소드마다 phase를 다시 뽑으므로
    (`SDN_DQN_new_env/network_env.py:256`) 외울 수 없다."

    외운 정책은 입력이 흔들려도 같은 길을 걷는다. 고장을 **아예 끄고** 서로 다른
    경로가 몇 개 나오는지, 최빈 경로가 몇 %를 덮는지 센다.
    """
    model = RLC.load_risk_model()
    pick = dqn_picker(model, next(model.parameters()).device)
    print("경로 다양성 — 고장 없음(churn=%.2f), %d에피소드 x %d시드"
          % (churn_p, episodes, len(SEEDS)))
    print("  %-11s %10s %12s %10s"
          % ("scenario", "서로다른길", "최빈 경로 비중", "phase수"))
    for scenario in SCENARIOS:
        seen = {}
        phases = set()
        for seed in SEEDS:
            for ep in range(episodes):
                random.seed(seed * 1000 + ep)
                np.random.seed(seed * 1000 + ep)
                env = make_env(scenario, churn_p, (seed, ep), "link")
                phases.add(env.dynamic_phase)
                walk(env, pick)
                if env.current_node == DESTINATION:
                    k = tuple(env.visited)
                    seen[k] = seen.get(k, 0) + 1
        tot = sum(seen.values())
        top = max(seen.values()) if seen else 0
        print("  %-11s %10d %11.1f%% %10d"
              % (scenario, len(seen), 100.0 * top / max(tot, 1), len(phases)))
    print("  최빈 비중이 100%에 가까우면 정책은 하나의 길을 외운 것이다 —")
    print("  그 경우 risk 입력을 흔들어도 행동이 안 바뀐다.")
    print()


def two_prop_p(k1, n1, k2, n2):
    """두 비율 차의 양측 검정(합동 분산 z). 짝이 아닌 비교에 쓴다."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / float(n1), k2 / float(n2)
    pp = (k1 + k2) / float(n1 + n2)
    se = math.sqrt(pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return float("nan")
    return math.erfc(abs(p1 - p2) / se / math.sqrt(2.0))


def measure_mask_work(episodes, churn_p=0.50, unit="node"):
    """끊긴 링크를 피하는 일을 누가 하는가 — 정책인가 행동 마스크인가.

    7-2절은 "표시를 꺼도 도달률이 같다"를 보여 준다. 그 해석으로 "마스크가 일을
    하고 risk 표시는 홉만 더한다"를 말하려면 **마스크가 실제로 개입하는지**를
    세어야 한다. 매 스텝 두 집합을 비교한다:

      물리 이웃(`EDGES` 기준, 재방문 제외) 대 살아 있는 이웃(`get_valid_actions`)

    차집합이 비어 있지 않은 스텝을 '기회'로 세고, 그 스텝에서 **마스크를 걷은**
    argmax가 죽은 링크를 고르면 '차단'으로 센다. 이동 자체는 정상 마스크 선택으로
    한다 — 궤적을 바꾸면 다른 실험이 된다.

    차단률이 높으면 정책은 끊긴 링크를 스스로 피하지 않는다. 표시를 켰을 때 그
    비율이 떨어지면 신호가 예견에 쓰이는 것이고, 안 떨어지면 쓰이지 않는 것이다.
    """
    import torch
    model = RLC.load_risk_model()
    device = next(model.parameters()).device
    phys = {i: set() for i in range(NUM_NODES)}
    for u, v in EDGES:
        phys[u].add(v)
        phys[v].add(u)

    print("마스크가 하는 일 (churn=%.2f, unit=%s, %d에피소드 x %d시드)"
          % (churn_p, unit, episodes, len(SEEDS)))
    print("  %-11s %-8s %9s %9s %9s %9s"
          % ("scenario", "표시", "전체스텝", "기회스텝", "차단", "차단률"))
    acc = {}
    for scenario in SCENARIOS:
        for mark in (True, False):
            steps = chance = blocked = 0
            for seed in SEEDS:
                for ep in range(episodes):
                    random.seed(seed * 1000 + ep)
                    np.random.seed(seed * 1000 + ep)
                    env = make_env(scenario, churn_p, (seed, ep), unit,
                                   mark_risk=mark)
                    while True:
                        live = [a for a in env.get_valid_actions()
                                if a not in env.visited]
                        if not live:
                            break
                        cur = env.current_node
                        allp = [a for a in phys[cur]
                                if a not in env.visited]
                        dead = set(allp) - set(live)
                        st = env.get_state()
                        with torch.no_grad():
                            q = model(torch.FloatTensor(st)
                                      .unsqueeze(0).to(device))[0]
                        steps += 1
                        if dead:
                            chance += 1
                            # 마스크를 걷되 **물리 이웃 안에서만** 고른다.
                            # 전체 25노드로 열면 인접하지 않은 노드가 뽑혀
                            # 마스크가 아닌 토폴로지가 막은 것을 세게 된다.
                            open_q = torch.full_like(q, -1e9)
                            for a in allp:
                                open_q[a] = q[a]
                            if int(torch.argmax(open_q).item()) in dead:
                                blocked += 1
                        masked = torch.full_like(q, -1e9)
                        for a in live:
                            masked[a] = q[a]
                        nxt = int(torch.argmax(masked).item())
                        _, _, done, _ = env.step(nxt)
                        if done or env.current_node == DESTINATION:
                            break
            acc[(scenario, mark)] = (steps, chance, blocked)
            rate = 100.0 * blocked / chance if chance else float("nan")
            print("  %-11s %-8s %9d %9d %9d %8.1f%%"
                  % (scenario if mark else "", "켬" if mark else "끔",
                     steps, chance, blocked, rate))
    print("  기회스텝 = 물리 이웃 중 죽은 링크가 하나라도 있던 스텝.")
    print("  차단률 = 그중 마스크를 걷은 argmax가 죽은 링크를 고른 비율.")
    print("  높으면 정책이 스스로 피하지 않는다는 뜻이다 — 마스크가 일을 한다.")
    print()
    print("  표시가 차단률을 낮추는가 (두 비율 검정, **짝이 아님**)")
    print("    %-11s %10s %10s %9s %9s"
          % ("scenario", "차단률 켬", "차단률 끔", "차이", "p"))
    for scenario in SCENARIOS:
        _, c1, b1 = acc[(scenario, True)]
        _, c0, b0 = acc[(scenario, False)]
        r1 = 100.0 * b1 / c1 if c1 else float("nan")
        r0 = 100.0 * b0 / c0 if c0 else float("nan")
        print("    %-11s %9.1f%% %9.1f%% %8.1fp %9s"
              % (scenario, r1, r0, r1 - r0,
                 fmt_p(two_prop_p(b1, c1, b0, c0))))
    print("    짝 검정이 아니다 — 두 팔은 서로 다른 곳에 서 있으므로 기회스텝이")
    print("    같은 상황이 아니다. 관찰 비교로만 읽어야 한다.")
    print("    Q값 자체의 한계: 학습 중에는 고장이 없었으므로 끊긴 링크의 Q는")
    print("    **살아 있던 링크로 학습된 값**이다. 그래서 모델이 '이 링크는 죽었다'를")
    print("    표현할 통로는 risk 입력 하나뿐이고, 차단률은 그 통로가 얼마나")
    print("    안 쓰이는지를 직접 재는 값이다.")
    print()


def risk_quantile(scenario, q=0.95, steps=20):
    """시나리오의 risk 분포 분위수. 고장 없는 환경에서 뽑는다."""
    vals = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        env = make_env(scenario, 0.0, (seed, 0))
        for _ in range(steps):
            env.update_dynamic_qos()
            for u, v in EDGES:
                vals.append(env.qos[env.normalize_edge(u, v)]["risk"])
    vals.sort()
    return vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]


def ablate_level(episodes, churn_p=0.50, unit="node"):
    """표시의 **크기**가 원인인가, 표시의 **존재**가 원인인가.

    7-1절에서 표시값 1.0은 `normal`의 risk 분포 밖으로 28.6 표준편차 나가 있다.
    그러면 경로 변화를 "정책이 위험을 읽었다"로 읽을 수 없다 — "분포 밖 입력에
    무너졌다"와 구분이 안 된다. 같은 고장 수열에서 세 팔을 돌려 가른다:

      A 표시 없음 · B 분포 **안**의 큰 값(시나리오별 95분위) · C 1.0

    A-B가 닮고 A-C가 갈리면 원인은 **크기**(분포 밖)다. A-B와 A-C가 비슷하게
    갈리면 원인은 **표시의 존재**이고, 그때만 "정책이 신호를 쓴다"고 말할 수 있다.
    """
    model = RLC.load_risk_model()
    device = next(model.parameters()).device
    pick = dqn_picker(model, device)
    print("절제 — 표시 크기 (churn=%.2f, unit=%s, %d에피소드 x %d시드)"
          % (churn_p, unit, episodes, len(SEEDS)))
    print("  A 표시없음 · B 95분위 · C %.1f" % DOWN_RISK)
    print("  %-11s %7s %9s %9s %8s %8s %8s"
          % ("scenario", "B값", "경로 A=B", "경로 A=C", "도달 A", "도달 B",
             "도달 C"))
    rows = []
    for scenario in SCENARIOS:
        p95 = risk_quantile(scenario)
        arms = (None, p95, DOWN_RISK)      # None = 표시 없음
        same = [0, 0]                      # [A=B, A=C]
        succ = [0, 0, 0]
        hop = [[], []]                     # [B-A, C-A], 양쪽 도달한 짝만
        # 엇갈린 짝 [A만 도달, 상대만 도달] — A 대비 B, A 대비 C
        disc = [[0, 0], [0, 0]]
        n_ok = 0
        for seed in SEEDS:
            for ep in range(episodes):
                paths, ok = [], []
                for j, mv in enumerate(arms):
                    random.seed(seed * 1000 + ep)
                    np.random.seed(seed * 1000 + ep)
                    env = make_env(scenario, churn_p, (seed, ep), unit,
                                   mark_risk=mv is not None, mark_value=mv)
                    walk(env, pick)
                    paths.append(tuple(env.visited))
                    ok.append(env.current_node == DESTINATION)
                    if ok[-1]:
                        succ[j] += 1
                n_ok += 1
                for i in (1, 2):
                    if paths[0] == paths[i]:
                        same[i - 1] += 1
                    if ok[0] and ok[i]:
                        hop[i - 1].append(len(paths[i]) - len(paths[0]))
                    if ok[0] and not ok[i]:
                        disc[i - 1][0] += 1
                    elif ok[i] and not ok[0]:
                        disc[i - 1][1] += 1
        rows.append((scenario, p95, same, succ, hop, n_ok, disc))
        print("  %-11s %7.3f %8.1f%% %8.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (scenario, p95,
                 100.0 * same[0] / max(n_ok, 1),
                 100.0 * same[1] / max(n_ok, 1),
                 100.0 * succ[0] / max(n_ok, 1),
                 100.0 * succ[1] / max(n_ok, 1),
                 100.0 * succ[2] / max(n_ok, 1)))
    print()
    print("  도달률 차 — 짝 검정 (A 기준)")
    print("  %-11s %14s %14s" % ("scenario", "A 대비 B", "A 대비 C"))
    for scenario, _p95, _same, _succ, _hop, _n, disc in rows:
        cells = ["%d/%-3d p=%s" % (d[0], d[1], fmt_p(mcnemar_p(d[0], d[1])))
                 for d in disc]
        print("  %-11s %14s %14s" % (scenario, cells[0], cells[1]))
    print("  b/c = A만 도달 / 상대만 도달 짝의 수")
    print()

    print("  홉 차이 (표시 없음 대비, 양쪽 모두 도달한 짝만)")
    print("  %-11s %14s %14s" % ("scenario", "B-A", "C-A"))
    for scenario, _p95, _same, _succ, hop, _n, _disc in rows:
        cells = []
        for h in hop:
            k = len(h)
            if k < 2:
                cells.append("표본부족")
                continue
            m = sum(h) / float(k)
            var = sum((x - m) ** 2 for x in h) / float(k - 1)
            cells.append("%+.2f+-%.2f (n=%d)" % (m, math.sqrt(var / k), k))
        print("  %-11s %14s %14s" % (scenario, cells[0], cells[1]))
    print("  A=B가 높고 A=C가 낮으면 경로 변화의 원인은 표시의 **크기**다")
    print()


def verify_churn_independent(steps=30):
    """경로가 달라도 같은 clock에서 같은 고장을 보는가.

    절제 실험의 인과 주장은 "표시만 바꿨다"에 전적으로 달렸다. 고장 추첨이 전역
    RNG를 쓰면 경로 길이가 갈린 뒤 고장 수열까지 갈려서, 경로 차이를 표시 탓으로
    돌릴 수 없게 된다. 코드를 읽어 `random.Random("seed-ep-clock")` 전용 인스턴스인
    것은 확인했지만 읽은 것은 주장이지 측정이 아니다. 일부러 **서로 다른 길**을 걷는
    두 정책(유효 행동 중 최소 번호 / 최대 번호)을 돌려 clock별 고장 집합을 맞춰 본다.
    """
    print("고장 독립성 — 다른 길을 걸어도 clock별 고장이 같은가")
    for unit in ("link", "node"):
        logs = []
        for pick in (min, max):
            random.seed(11)
            np.random.seed(11)
            env = make_env("unstable", 0.50, (0, 0), unit)
            log, path = [], [env.current_node]
            for _ in range(steps):
                valid = [a for a in env.get_valid_actions()
                         if a not in env.visited]
                if not valid:
                    break
                env.step(pick(valid))
                log.append((env.clock, frozenset(env.down.items())))
                path.append(env.current_node)
                if env.current_node == DESTINATION:
                    break
            logs.append((log, path))
        (la, pa), (lb, pb) = logs
        if pa == pb:
            raise SystemExit("고장 독립성 검증 불가 — 두 정책이 같은 길을 걸었다")
        for (ca, da), (cb, db) in zip(la, lb):
            if ca != cb or da != db:
                raise SystemExit(
                    "고장 독립성 실패 — clock %d에서 고장 집합이 갈린다" % ca)
        print("  %-5s 다른 경로(%d홉 대 %d홉) · clock별 고장 집합 %d개 일치 → 통과"
              % (unit, len(pa) - 1, len(pb) - 1, min(len(la), len(lb))))
    print("  통과: 고장은 clock만의 함수다 — 경로 차이를 표시 탓으로 돌릴 수 있다")
    print()


def risk_distribution(steps=20):
    """시나리오별 risk 분포. 모델 불필요.

    절제 실험에서 표시 효과가 `normal`에서만 컸다. 가설은 "normal은 risk가 전반적
    으로 낮아 고장 표시값 1.0이 분포에서 멀리 떨어진 교란이고, 이미 risk가 높은
    시나리오에서는 상대적으로 덜 흔든다"다. 추측으로 두지 않고 분포를 직접 본다.
    """
    print("risk 분포 — 고장 표시값 %.1f이 기존 분포에서 얼마나 떨어져 있는가"
          % DOWN_RISK)
    print("  %-11s %7s %7s %8s %8s %7s %9s"
          % ("scenario", "평균", "표준편차", ">0.5", ">0.9", "최대", "(1.0-평균)/SD"))
    for scenario in SCENARIOS:
        vals = []
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            env = make_env(scenario, 0.0, (seed, 0))
            for _ in range(steps):
                env.update_dynamic_qos()
                for u, v in EDGES:
                    vals.append(env.qos[env.normalize_edge(u, v)]["risk"])
        m, sd = mean_std(vals)
        hi5 = 100.0 * sum(1 for v in vals if v > 0.5) / len(vals)
        hi9 = 100.0 * sum(1 for v in vals if v > 0.9) / len(vals)
        z = (DOWN_RISK - m) / sd if sd else float("nan")
        print("  %-11s %7.3f %7.3f %7.1f%% %7.1f%% %7.3f %9.2f"
              % (scenario, m, sd, hi5, hi9, max(vals), z))
    print("  z가 클수록 표시가 학습 분포에서 먼 입력이다 — 교란도 그만큼 크다")
    print()


def mcnemar_p(b, c):
    """짝지은 이진 결과의 정확 검정. b/c는 엇갈린 짝의 수다.

    같은 고장 수열에서 한쪽만 실패한 에피소드만 정보를 준다 — 둘 다 성공하거나
    둘 다 실패한 짝은 두 조건을 가르지 못한다.
    """
    m = b + c
    if m == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / float(2 ** m)
    return min(1.0, 2.0 * tail)


def ablate_mark(episodes, churn_p=0.50, unit="node"):
    """위험 표시를 끄면 DQN이 다른 길을 걷는가.

    양성 대조는 신호가 **입력에 있다**는 것만 증명한다. 정책이 그 신호를 쓰는지는
    별개다 — `risk=1.0`은 학습 분포 밖의 값이고(학습 중 risk는 공식의 연속 출력),
    불법 이동을 막는 것은 행동 마스크다. 같은 고장 수열에서 표시만 끄고 경로가
    그대로면, 제안 모델은 이탈을 **예견**하는 기제가 없고 이미 못 쓰게 된 링크에
    마스크로 반응하는 것뿐이다. 그 경우 C-1의 널은 환경 탓이 아니라 모델의
    설계 한계다.
    """
    import torch
    model = RLC.load_risk_model()
    device = next(model.parameters()).device
    print("절제 — 위험 표시 끄기 (churn=%.2f, unit=%s, %d에피소드)"
          % (churn_p, unit, episodes))
    rows = []
    for scenario in SCENARIOS:
        same, n_ok, hops = 0, 0, []
        succ = [0, 0]
        disc = [0, 0]       # [표시만 실패, 무표시만 실패]
        for seed in SEEDS:
            for ep in range(episodes):
                paths, ok = [], []
                for j, mark in enumerate((True, False)):
                    random.seed(seed * 1000 + ep)
                    np.random.seed(seed * 1000 + ep)
                    env = make_env(scenario, churn_p, (seed, ep), unit,
                                   mark_risk=mark)

                    def next_node_of(e):
                        valid = [a for a in e.get_valid_actions()
                                 if a not in e.visited]
                        if not valid:
                            return None
                        st = e.get_state()
                        with torch.no_grad():
                            q = model(torch.FloatTensor(st)
                                      .unsqueeze(0).to(device))[0]
                        masked = torch.full_like(q, -1e9)
                        for a in valid:
                            masked[a] = q[a]
                        return int(torch.argmax(masked).item())

                    walk(env, next_node_of)
                    paths.append(tuple(env.visited))
                    ok.append(env.current_node == DESTINATION)
                    if ok[-1]:
                        succ[j] += 1
                n_ok += 1
                if paths[0] == paths[1]:
                    same += 1
                # 홉 차이는 **양쪽 모두 도달한** 에피소드만 쓴다. 실패하면
                # 경로가 잘려 짧게 잡혀 평균이 거짓으로 움직인다.
                if ok[0] and ok[1]:
                    hops.append(len(paths[0]) - len(paths[1]))
                if ok[0] and not ok[1]:
                    disc[1] += 1
                elif ok[1] and not ok[0]:
                    disc[0] += 1
        rows.append((scenario, same, n_ok, succ, disc, hops))

    print("  (가) 경로가 바뀌는가 · 도달률이 달라지는가")
    print("    %-11s %8s %10s %12s %9s %8s"
          % ("scenario", "경로동일", "도달(표시)", "도달(무표시)",
             "짝 b/c", "도달 p"))
    for scenario, same, n_ok, succ, disc, _hops in rows:
        print("    %-11s %7.1f%% %9.1f%% %11.1f%% %5d/%-3d %8s"
              % (scenario, 100.0 * same / max(n_ok, 1),
                 100.0 * succ[0] / max(n_ok, 1),
                 100.0 * succ[1] / max(n_ok, 1),
                 disc[0], disc[1], fmt_p(mcnemar_p(disc[0], disc[1]))))
    print("    짝 b/c = 표시만 실패 / 무표시만 실패 짝의 수. 도달 p는 그 둘의")
    print("    정확 검정(McNemar)이다 — 짝지은 이진 결과는 엇갈린 짝만 정보를 준다.")
    print()

    print("  (나) 표시된 경로가 더 긴가 — **양쪽 모두 도달한** 짝만")
    print("    %-11s %12s %7s %12s %9s"
          % ("scenario", "홉차 평균±SE", "n", "길어짐/짧아짐", "부호 p"))
    for scenario, _same, _n_ok, _succ, _disc, hops in rows:
        k = len(hops)
        if k == 0:
            print("    %-11s %12s" % (scenario, "표본 없음"))
            continue
        m = sum(hops) / float(k)
        if k > 1:
            var = sum((h - m) ** 2 for h in hops) / float(k - 1)
            se = math.sqrt(var / k)
        else:
            se = float("nan")
        up = sum(1 for h in hops if h > 0)
        dn = sum(1 for h in hops if h < 0)
        print("    %-11s %+7.2f±%.2f %7d %6d/%-5d %9s"
              % (scenario, m, se, k, up, dn, fmt_p(mcnemar_p(up, dn))))
    print("    부호 검정을 쓴다 — 홉차는 작은 정수값이라 정규 근사보다 안전하다.")
    print("    (+)는 위험 표시를 켠 경로가 더 길다는 뜻이다.")
    print()


DECODE_RULES = (("전체배제(현행)", "all"), ("직전만", "prev"),
                ("배제없음", "none"))


def blocked_by(env, rule):
    """디코딩 규칙이 막는 노드 집합. 'all'이 본 실험에서 쓴 현행 규칙이다.

    원본 `step()`은 재방문을 **보상으로만** 벌한다(network_env.py:367, -8.0) —
    이동 자체는 실행된다. 즉 재방문을 금지하는 것은 원본 환경이 아니라 내
    래퍼뿐이므로, 규칙을 풀어도 원본 코드는 건드리지 않는다.
    """
    if rule == "all":
        return set(env.visited)
    if rule == "prev":
        return {env.visited[-2]} if len(env.visited) >= 2 else set()
    return set()


def probe_decode(episodes, churn_p=0.50, unit="node_protected"):
    """도달률 결손이 라우팅 탓인가 디코딩 규칙 탓인가.

    보호 조건에서는 고립·경로없음이 셋 다 0이므로 결손 = 디코딩 실패 차다(이건
    항등식이지 측정이 아니다). 갈라야 할 것은 그 디코딩 실패가 **규칙이 만든
    인공물**인지 **정책에 복구 동작이 없다는 사실**인지다. 규칙을 풀어서
    에피소드가 돌아오면 인공물이고, 핑퐁하다 걸음수 한계에 걸리면 정책 쪽이다.
    """
    import torch
    model = RLC.load_risk_model()
    device = next(model.parameters()).device
    print("디코딩 규칙 절제 (%s, churn=%.2f, %d에피소드 x %d시드 x %d시나리오)"
          % (unit, churn_p, episodes, len(SEEDS), len(SCENARIOS)))
    print("  %-18s %-14s %7s %7s %7s %8s %6s %6s"
          % ("방법", "규칙", "도달", "고립", "디코딩", "걸음한계", "걸음", "노드"))
    for variant, cost in VARIANTS:
        for label, rule in DECODE_RULES:
            if cost is None:
                def pick(e, _r=rule):
                    bad = blocked_by(e, _r)
                    valid = [a for a in e.get_valid_actions()
                             if a not in bad]
                    if not valid:
                        return None
                    st = e.get_state()
                    with torch.no_grad():
                        q = model(torch.FloatTensor(st)
                                  .unsqueeze(0).to(device))[0]
                    m = torch.full_like(q, -1e9)
                    for a in valid:
                        m[a] = q[a]
                    return int(torch.argmax(m).item())
            else:
                def pick(e, _r=rule, _c=cost):
                    path = plan_live(e, blocked_by(e, _r), _c)
                    return path[1] if len(path) >= 2 else None

            succ, cnt, hops, reasons = 0, 0, [], []
            dist = []
            for seed in SEEDS:
                for sc in SCENARIOS:
                    for ep in range(episodes):
                        random.seed(seed * 1000 + ep)
                        np.random.seed(seed * 1000 + ep)
                        env = make_env(sc, churn_p, (seed, ep), unit)
                        _, reason = walk(env, pick)
                        cnt += 1
                        reasons.append(reason)
                        if env.current_node == DESTINATION:
                            succ += 1
                            hops.append(len(env.visited))
                            dist.append(len(set(env.visited)))
            d = max(cnt, 1)
            print("  %-18s %-14s %6.1f%% %6.1f%% %6.1f%% %7.1f%% %6.2f %6.2f"
                  % (variant, label, 100.0 * succ / d,
                     100.0 * reasons.count("stuck_isolated") / d,
                     100.0 * reasons.count("stuck_decode") / d,
                     100.0 * reasons.count("step_limit") / d,
                     sum(hops) / max(len(hops), 1),
                     sum(dist) / max(len(dist), 1)))
    print("  걸음 = 재방문을 포함한 이동 횟수, 노드 = 서로 다른 노드 수.")
    print("  규칙을 풀면 (ㄱ) 플래너가 쓸 수 있는 그래프가 넓어져 경로가 짧아지고")
    print("  (ㄴ) 걸어온 길을 되밟아 걸음이 늘 수 있다 — 부호가 반대인 두 통로다.")
    print("  노드 수가 줄면 (ㄱ), 걸음만 늘면 (ㄴ)이다.")
    print("  '배제없음'에서 도달률이 안 오르고 걸음한계만 늘면, 규칙이 실패를")
    print("  제조한 것이 아니라 정책에 되돌아가는 복구 동작이 없는 것이다.")


def trace_walk(env, pick):
    """걸으면서 **clock별로 내려가 있는 간선 집합**을 적는다.

    측정값을 바꾸지 않는 관찰 전용 보행이다 — `walk()`과 같은 순서로 돌고
    실현 QoS만 안 모은다. 교란 점검은 "같은 clock에 같은 고장인가"만
    물으므로 clock을 열쇠로 찍어둔다.
    """
    trace = {env.clock: tuple(sorted(env.down))}
    done = False
    while not done:
        nxt = pick(env)
        if nxt is None:
            break
        _, _, done, info = env.step(nxt)
        trace[env.clock] = tuple(sorted(env.down))
        if isinstance(info, dict) and info.get("reason") == "invalid":
            break
        if len(env.visited) >= MAX_STEPS:
            break
    return trace, len(env.visited)


def make_picker(model, device, rule, cost):
    """다음 노드를 고르는 함수. cost가 None이면 정책, 아니면 재계획 기준선."""
    import torch

    if cost is None:
        def pick(e):
            bad = blocked_by(e, rule)
            valid = [a for a in e.get_valid_actions() if a not in bad]
            if not valid:
                return None
            st = e.get_state()
            with torch.no_grad():
                q = model(torch.FloatTensor(st)
                          .unsqueeze(0).to(device))[0]
            m = torch.full_like(q, -1e9)
            for a in valid:
                m[a] = q[a]
            return int(torch.argmax(m).item())
        return pick

    def pick(e):
        path = plan_live(e, blocked_by(e, rule), cost)
        return path[1] if len(path) >= 2 else None
    return pick


def seq_agree(tr_a, tr_b):
    """공통 clock에서 고장 집합이 같은가. 같으면 None, 다르면 첫 clock."""
    common = set(tr_a) & set(tr_b)
    diff = sorted(c for c in common if tr_a[c] != tr_b[c])
    return diff[0] if diff else None


def probe_agree_subset(episodes, churn_p=0.50, unit="node_protected"):
    """P1 결손을 **같은 고장을 본 에피소드만**으로 다시 검정한다.

    `--probe-cross`가 센 것: `node_protected`에서는 방법 간 고장 수열이
    3분의 1 가량 갈린다(`link`·`node`는 100% 동일). P1의 Holm 생존 결손
    두 건이 바로 그 조건의 방법 간 비교이므로, 짝의 전제가 깨진 셀이
    섞여 있다. 그래서 수열이 **일치한 에피소드만** 남겨 같은 검정을
    다시 돌린다.

    읽기는 비대칭이다. 결손이 **남으면** 같은 고장 아래서도 나는 차이이므로
    결손 쪽 증거가 강해진다. 결손이 **사라지면** 그 차이가 갈린 셀에만
    업혀 있었다는 뜻인데, 그것만으로 교란이라 단정할 수는 없다 — 일치
    여부는 두 보행이 같았는지로 정의되므로 **처치 후 선택**이고, 홉이 긴
    방법이 더 오래 노출된다는 효과(r(홉, 실패)=+0.486)와도 섞인다.
    """
    model = RLC.load_risk_model()
    device = next(model.parameters()).device

    data = {}
    for variant, cost in VARIANTS:
        pick = make_picker(model, device, DECODE_RULE, cost)
        for seed in SEEDS:
            for sc in SCENARIOS:
                for ep in range(episodes):
                    random.seed(seed * 1000 + ep)
                    np.random.seed(seed * 1000 + ep)
                    env = make_env(sc, churn_p, (seed, ep), unit)
                    tr, _ = trace_walk(env, pick)
                    ok = 1.0 if env.current_node == DESTINATION else 0.0
                    data[(variant, seed, sc, ep)] = (tr, ok)

    names = [v for v, _ in VARIANTS]
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            all_d, agr_d, kept, tot = [], [], 0, 0
            for seed in SEEDS:
                for sc in SCENARIOS:
                    sa, sb, na = 0.0, 0.0, 0
                    ga, gb, ng = 0.0, 0.0, 0
                    for ep in range(episodes):
                        ta, ra = data[(a, seed, sc, ep)]
                        tb, rb = data[(b, seed, sc, ep)]
                        sa, sb, na = sa + ra, sb + rb, na + 1
                        tot += 1
                        if seq_agree(ta, tb) is None:
                            ga, gb, ng = ga + ra, gb + rb, ng + 1
                            kept += 1
                    if na:
                        all_d.append(100.0 * (sa - sb) / na)
                    if ng:
                        agr_d.append(100.0 * (ga - gb) / ng)
            out.append([a, b, all_d, agr_d, kept, tot])

    print("P1 재검정 — 같은 고장만 (%s, churn=%.2f, 규칙=%s)"
          % (unit, churn_p, DECODE_RULE))
    print("  %d에피소드 x %d시드 x %d시나리오, 셀 = (시나리오, 시드)"
          % (episodes, len(SEEDS), len(SCENARIOS)))
    pa = holm_bonferroni([paired_t(x[2])[2] for x in out])
    pg = holm_bonferroni([paired_t(x[3])[2] for x in out])
    print("  %-34s %17s %17s %7s"
          % ("대조", "전체 셀", "일치 셀만", "일치율"))
    for x, qa, qg in zip(out, pa, pg):
        ma = paired_t(x[2])[0]
        mg = paired_t(x[3])[0]
        print("  %-34s %+6.2f%%p %-9s %+6.2f%%p %-9s %6.1f%%"
              % ("%s - %s" % (x[0], x[1]), ma, fmt_p(qa), mg, fmt_p(qg),
                 100.0 * x[4] / max(x[5], 1)))
    print("  값 = 짝 차의 평균(%%p), p = Holm 보정(비교 %d번)." % len(out))
    print("  일치율 = 두 방법이 같은 고장을 본 에피소드 비율.")
    print("  결손이 '일치 셀만'에서도 살면 같은 고장 아래의 차이다.")
    print("  사라지면 갈린 셀에 업혀 있었다는 뜻이지만, 일치 여부는")
    print("  처치 후에 정해지므로 그것만으로 교란이라 단정하지 않는다.")


def probe_churn_cross(episodes, churn_p=0.50):
    """**방법 사이**의 짝 검정 전제를 점검한다.

    5절·6절의 도달률 검정은 "세 방법이 같은 `(seed, ep)` 고장을 본다"를 근거로
    (시나리오, 시드) 짝으로 묶는다. 고장이 `(seed, ep, clock)`만의 함수이므로
    `link`·`node`에서는 참이다. 그런데 `node_protected`는 뽑기 전에 후보에서
    **서 있는 노드**를 뺀다(188행). 방법마다 서 있는 자리가 다르므로 같은
    clock에 **다른 노드**가 떠난다 — 그 조건에서는 짝의 근거가 약해진다.

    P1의 Holm 생존 결손 두 건(-2.17 %p, -1.83 %p)이 바로 `node_protected`
    0.50의 방법 간 비교다. 그래서 그 약함을 추정하지 않고 센다. `link`와
    `node`가 '동일 100.0%'로 나와야 하고, 안 나오면 위의 읽기가 틀린 것이다.
    """
    model = RLC.load_risk_model()
    device = next(model.parameters()).device

    print("방법 간 고장 수열 점검 (규칙=%s, churn=%.2f, %d에피 x %d시드 x %d시나)"
          % (DECODE_RULE, churn_p, episodes, len(SEEDS), len(SCENARIOS)))
    print("  %-15s %-34s %8s %7s"
          % ("단위", "대조", "수열동일", "첫갈림"))
    for unit in CHURN_UNITS:
        traces = {}
        for variant, cost in VARIANTS:
            pick = make_picker(model, device, DECODE_RULE, cost)
            for seed in SEEDS:
                for sc in SCENARIOS:
                    for ep in range(episodes):
                        random.seed(seed * 1000 + ep)
                        np.random.seed(seed * 1000 + ep)
                        env = make_env(sc, churn_p, (seed, ep), unit)
                        tr, _ = trace_walk(env, pick)
                        traces[(variant, seed, sc, ep)] = tr
        names = [v for v, _ in VARIANTS]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                same, firsts, tot = 0, [], 0
                for seed in SEEDS:
                    for sc in SCENARIOS:
                        for ep in range(episodes):
                            ta = traces[(a, seed, sc, ep)]
                            tb = traces[(b, seed, sc, ep)]
                            tot += 1
                            c = seq_agree(ta, tb)
                            if c is None:
                                same += 1
                            else:
                                firsts.append(c)
                d = max(tot, 1)
                first = "-"
                if firsts:
                    first = "%.1f" % (sum(firsts) / float(len(firsts)))
                print("  %-15s %-34s %7.1f%% %7s"
                      % (unit, "%s vs %s" % (a, b), 100.0 * same / d, first))
    print("  수열동일 = 공통 clock에서 내려간 간선 집합이 전부 같은 비율.")
    print("  link/node는 대조군이다 - 100.0%가 아니면 고장이 경로에 의존한다.")
    print("  node_protected가 낮으면, 그 조건의 방법 간 짝 검정은 '같은 고장")
    print("  아래 두 정책'이 아니라 '각자 다른 고장을 본 두 정책'의 비교다.")


def probe_churn_sync(episodes, churn_p=0.50):
    """규칙을 바꾼 두 실행이 **같은 고장**을 보는가.

    고장은 `random.Random("seed-ep-clock")`에서 뽑으므로(172행) 전역 RNG와
    무관하고 걸어온 길에도 무관하다 — `link`(196행)와 `node`(184행)에서는
    그렇다. 그런데 `node_protected`는 뽑기 전에 후보에서 **서 있는 노드**를
    뺀다(188행). 규칙을 풀면 걸음이 달라지고, 걸음이 달라지면 후보 목록이
    달라지고, 그러면 같은 clock에 **다른 노드**가 떠난다. 그 조건에서는
    "규칙만 바꾼 짝 비교"라는 6-2절의 전제가 약해진다.

    이 진단은 그 약함을 주장하지 않고 센다. `link`와 `node`는 대조군이므로
    '동일 100.0%'가 나와야 하고, 안 나오면 위의 읽기가 틀린 것이다.
    걸음 수가 달라지는 것 자체는 교란이 아니라 규칙의 효과다 — 교란은
    **같은 clock에 서로 다른 고장**이 뜨는 것뿐이므로 공통 clock만 본다.
    """
    model = RLC.load_risk_model()
    device = next(model.parameters()).device

    def picker(rule, cost):
        return make_picker(model, device, rule, cost)

    def one_pass(unit, cost, rule):
        out = {}
        for seed in SEEDS:
            for sc in SCENARIOS:
                for ep in range(episodes):
                    random.seed(seed * 1000 + ep)
                    np.random.seed(seed * 1000 + ep)
                    env = make_env(sc, churn_p, (seed, ep), unit)
                    out[(seed, sc, ep)] = trace_walk(env, picker(rule, cost))
        return out

    print("고장 수열 동기 점검 (churn=%.2f, %d에피소드 x %d시드 x %d시나리오)"
          % (churn_p, episodes, len(SEEDS), len(SCENARIOS)))
    print("  %-18s %-15s %-10s %8s %9s %7s"
          % ("방법", "단위", "대조", "수열동일", "걸음동일", "첫갈림"))
    for unit in CHURN_UNITS:
        for variant, cost in VARIANTS:
            base = one_pass(unit, cost, "all")
            for label, rule in (("직전만", "prev"), ("배제없음", "none")):
                cur = one_pass(unit, cost, rule)
                same, steps_same, firsts = 0, 0, []
                for key, (tr, steps) in cur.items():
                    tr0, steps0 = base[key]
                    if steps == steps0:
                        steps_same += 1
                    c = seq_agree(tr, tr0)
                    if c is None:
                        same += 1
                    else:
                        firsts.append(c)
                d = max(len(cur), 1)
                first = "-"
                if firsts:
                    first = "%.1f" % (sum(firsts) / float(len(firsts)))
                print("  %-18s %-15s %-10s %7.1f%% %8.1f%% %7s"
                      % (variant, unit, label, 100.0 * same / d,
                         100.0 * steps_same / d, first))
    print("  수열동일 = 공통 clock에서 내려간 간선 집합이 전부 같은 비율.")
    print("  걸음동일 = 걸음 수까지 같은 비율(이건 규칙의 효과이므로 낮아도")
    print("  된다). 첫갈림 = 처음 달라진 clock의 평균.")
    print("  link/node가 100.0%가 아니면 고장 수열이 경로에 의존한다는")
    print("  뜻이고, 그러면 6-2절의 짝 비교를 그대로 읽을 수 없다.")


def verify(episodes=20):
    """churn_p=0이 평범한 NetworkEnv를 경로까지 재현하는지 본다.

    재현되지 않으면 서브클래스 자체가 환경을 바꾼 것이므로, 뒤의 모든 수치가
    이동성이 아니라 내 구현 차이를 재게 된다. 그래서 측정 전에 멈춘다.
    """
    print("검증 — churn_p=0이 원본 환경과 같은가")
    bad = 0
    for scenario in SCENARIOS:
        paths_a, paths_b = [], []
        for tag, factory in (("plain", NetworkEnv), ("mobile", None)):
            random.seed(7)
            np.random.seed(7)
            for ep in range(episodes):
                if factory is None:
                    env = MobileEnv(churn_p=0.0, churn_key=(7, ep),
                                    use_risk=True, scenario=scenario,
                                    risk_window=RLC.RISK_WINDOW)
                else:
                    env = factory(use_risk=True, scenario=scenario,
                                  risk_window=RLC.RISK_WINDOW)
                env.reset()
                for _ in range(WARMUP):
                    env.update_dynamic_qos()

                def next_node_of(e):
                    path = plan_live(e, set(e.visited), "bandwidth")
                    return path[1] if len(path) >= 2 else None

                walk(env, next_node_of)
                (paths_a if factory is not None else paths_b).append(
                    tuple(env.visited))
        same = paths_a == paths_b
        print("  %-11s 경로 일치 %s (%d 에피소드)"
              % (scenario, "예" if same else "아니오", episodes))
        if not same:
            bad += 1
            for i, (a, b) in enumerate(zip(paths_a, paths_b)):
                if a != b:
                    print("    첫 불일치 ep%d\n      원본 %s\n      서브 %s"
                          % (i, a, b))
                    break
    if bad:
        raise SystemExit("검증 실패 — 서브클래스가 churn_p=0에서 환경을 바꿨다")
    print("  통과: 서브클래스는 고장이 없을 때 환경을 바꾸지 않는다\n")


# --------------------------------------------------------------- 측정


def measure(episodes):
    model = RLC.load_risk_model()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    fields = (["scenario", "variant", "churn_unit", "churn_p", "seed",
               "episodes", "success"] + list(REASONS)
              + ["rejected_per_ep", "mean_down_links"]
              + list(METRICS)
              + ["paired_n"] + ["paired_" + m for m in METRICS])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for scenario in SCENARIOS:
            for churn_p in CHURN_PS:
                # churn_p=0이면 단위가 뜻이 없다 — 한 번만 돈다.
                units = ("link",) if churn_p == 0.0 else CHURN_UNITS
                for unit in units:
                    for seed in SEEDS:
                        # 세 방법을 모두 돌린 뒤 공통 도달 집합을 만든다.
                        got = []
                        for variant, cost_type in VARIANTS:
                            if cost_type is None:
                                got.append((variant,) + run_dqn(
                                    scenario, seed, episodes, churn_p,
                                    model, unit))
                            else:
                                got.append((variant,) + run_ospf(
                                    scenario, seed, episodes, churn_p,
                                    cost_type, unit))
                        mask = [all(p[i] is not None for _, _, p in got)
                                for i in range(episodes)]
                        for variant, res, per_ep in got:
                            res.update(paired_metrics(per_ep, mask))
                            row = {"scenario": scenario, "variant": variant,
                                   "churn_unit": unit, "churn_p": churn_p,
                                   "seed": seed, "episodes": episodes}
                            row.update({k: res.get(k) for k in fields
                                        if k not in row})
                            writer.writerow(row)
                    fh.flush()
                    print("  %-11s churn=%.2f unit=%-4s 완료"
                          % (scenario, churn_p, unit), flush=True)
    print("기록: %s" % OUT_CSV)


# --------------------------------------------------------------- 보고


def load_rows(path=None):
    with open(path or OUT_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["churn_p"] = float(r["churn_p"])
        r["seed"] = int(r["seed"])
        for k, v in list(r.items()):
            if k in ("scenario", "variant", "churn_unit", "seed", "churn_p"):
                continue
            r[k] = float(v) if v not in ("", "nan", None) else float("nan")
    return rows


def per_seed(rows, unit, p, variant, field, scenario=None):
    out = [r[field] for r in rows
           if r["churn_unit"] == unit and r["churn_p"] == p
           and r["variant"] == variant
           and (scenario is None or r["scenario"] == scenario)]
    return [v for v in out if v == v]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def cell_map(rows, unit, p, variant, field):
    """(시나리오, 시드) -> 값. 짝 검정의 열쇠를 만든다."""
    out = {}
    for r in rows:
        if (r["churn_unit"] == unit and r["churn_p"] == p
                and r["variant"] == variant and r[field] == r[field]):
            out[(r["scenario"], r["seed"])] = r[field]
    return out


def paired_diff(rows, field, a, b):
    """a, b = (unit, churn_p, variant). 양쪽에 다 있는 열쇠만 뺀다."""
    ma = cell_map(rows, a[0], a[1], a[2], field)
    mb = cell_map(rows, b[0], b[1], b[2], field)
    return [ma[k] - mb[k] for k in sorted(set(ma) & set(mb))]


def paired_t(d):
    """차 목록 -> (평균, MDE, p). 같은 셀끼리 뺐으므로 한 표본 t다."""
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    m, sd = mean_std(d)
    se = sd / math.sqrt(len(d))
    if se == 0.0:
        return m, 0.0, (0.0 if m else float("nan"))
    df = len(d) - 1
    return m, t_critical(df) * se, t_two_sided_p(m / se, df)


def cross_diff(rows_a, rows_b, field, key):
    """두 CSV 사이의 짝 차. key=(unit, churn_p, variant), a 빼기 b.

    6-2절에서 규칙 하나만 바꾼 두 실행을 같은 (시나리오, 시드) 셀끼리 뺀다.
    표 평균끼리 빼면 시드 분산이 그대로 남지만, 셀끼리 빼면 규칙 효과만 남는다.
    """
    ma = cell_map(rows_a, key[0], key[1], key[2], field)
    mb = cell_map(rows_b, key[0], key[1], key[2], field)
    return [ma[k] - mb[k] for k in sorted(set(ma) & set(mb))]


def compare_rules():
    """6-2절: 평가 래퍼의 재방문 규칙을 바꾼 전면 재측정을 대조한다.

    6-1절은 한 조건(node_protected, churn=0.50)만 쟀다. 이 절은 규칙만 바꿔
    7조건 전체를 다시 돌린 CSV를 같은 (시나리오, 시드) 셀끼리 뺀다.
    """
    pa = os.path.join(BASE_DIR, "review_results", "mobility_proxy.csv")
    if not os.path.exists(pa):
        print("없음: %s" % pa)
        return
    ra = load_rows(pa)
    print("6-2) 디코딩 규칙 전면 대조 — 기준은 본 측정(전체배제)")
    print("   이 논문에 등장하는 재방문 규칙은 셋이고, 본 측정은 그중 가장")
    print("   가혹한 것을 썼다:")
    print("     배제없음   = **원본 평가**가 쓰는 규칙(run_policy.py,")
    print("                  risk_dqn_every_scenario.py는 이웃 전체에서")
    print("                  argmax를 고른다). 재방문은 학습 때 보상으로만")
    print("                  벌했다(network_env.py:367, -8.0).")
    print("     미방문우선 = R3-4의 변형(reeval_generalization_conventions")
    print("                  .py:111). 미방문 이웃이 있으면 그 중에서만,")
    print("                  없으면 전체 — 하드 실패가 없다.")
    print("     전체배제   = C-1 본 측정에 쓴 **내 래퍼**의 규칙. 이웃을 전부")
    print("                  밟았으면 실패(stuck_decode)로 적는다.")
    print("   즉 본 측정의 디코딩 실패는 원본 평가에는 없는 실패 유형이다.")
    found = 0
    for label, suffix in (("직전만", "_prev"), ("배제없음(원본)", "_none")):
        pb = os.path.join(BASE_DIR, "review_results",
                          "mobility_proxy%s.csv" % suffix)
        if not os.path.exists(pb):
            print()
            print("   [%s] CSV 없음 — %s" % (label, os.path.basename(pb)))
            continue
        found += 1
        print()
        compare_one(ra, load_rows(pb), label)
    if not found:
        print("   비교할 CSV가 없다.")


def compare_one(ra, rb, label):
    """본 측정(ra)과 규칙을 바꾼 재측정(rb)을 7조건 전체에서 뺀다."""

    def key5(r):
        return (r["churn_unit"], r["churn_p"], r["variant"],
                r["scenario"], r["seed"])

    ka = {key5(r) for r in ra}
    kb = {key5(r) for r in rb}
    common = ka & kb
    print("   === 전체배제 대 %s" % label)
    print("   셀 수: 본 측정 %d · %s %d · 공통 %d"
          % (len(ka), label, len(kb), len(common)))
    if len(kb) < len(ka):
        print("   %s 쪽이 %d셀 모자라므로 아래는 **부분 대조**다(진행 중)."
              % (label, len(ka) - len(kb)))
    eps = {r["episodes"] for r in ra} | {r["episodes"] for r in rb}
    print("   에피소드 수 집합 %s — 다르면 짝 검정을 쓸 수 없다."
          % sorted(eps))
    hop_a = {key5(r): r["hop"] for r in ra}
    hop_b = {key5(r): r["hop"] for r in rb}
    zero = [k for k in common if k[1] == 0.0]
    zdiff = [k for k in zero if abs(hop_a[k] - hop_b[k]) > 1e-9]
    print("   고장 없음(churn=0.00) 공통 셀 %d개 중 홉이 바뀐 셀 %d개"
          % (len(zero), len(zdiff)))
    print("   — 0개면 규칙은 고장이 났을 때만 문다는 뜻이다.")

    def pair_means(field, unit, p, nm):
        """공통 셀만 쓴 두 평균과 셀 수. 부분 대조에서도 차가 두 평균의 차와

        같아지도록 양쪽을 같은 열쇠 집합으로 자른다.
        """
        ma = cell_map(ra, unit, p, nm, field)
        mb = cell_map(rb, unit, p, nm, field)
        ks = sorted(set(ma) & set(mb))
        if not ks:
            return None
        return (sum(ma[k] for k in ks) / len(ks),
                sum(mb[k] for k in ks) / len(ks), len(ks))

    units = [u for u in CHURN_UNITS
             if any(r["churn_unit"] == u for r in rb)]
    ps = sorted({r["churn_p"] for r in rb if r["churn_p"] > 0})
    names = [v for v, _ in VARIANTS]

    print()
    print("   도달률: 본 측정 -> %s (짝 차, Holm 보정)" % label)
    print("   %-15s %-6s %-18s %7s %7s %8s %8s"
          % ("unit", "churn", "variant", "본측정", label, "차(%p)", "보정p"))
    qr = []
    for unit in units:
        for p in ps:
            for nm in names:
                d = cross_diff(rb, ra, "success", (unit, p, nm))
                if len(d) < 3:
                    continue
                pm = pair_means("success", unit, p, nm)
                m, mde, pv = paired_t(d)
                qr.append((unit, p, nm, pm[0], pm[1], m, mde, pv))
    if not qr:
        print("   (고장 조건 데이터가 아직 없다)")
        return
    for x, aq in zip(qr, holm_bonferroni([z[7] for z in qr])):
        print("   %-15s %-6.2f %-18s %6.1f%% %6.1f%% %+8.2f %8s%s"
              % (x[0], x[1], x[2], x[3], x[4], x[5], fmt_p(aq),
                 "" if not (aq == aq and aq < 0.05) else " *"))
    print("   비교 %d번 Holm 보정. 별표는 규칙을 바꾸니 도달률이 유의하게"
          " 달라졌다는 뜻이다." % len(qr))

    print()
    print("   디코딩 실패와 걸음수 한계 실패 (A=본 측정, B=%s)" % label)
    print("   %-15s %-6s %-18s %9s %9s %9s %9s"
          % ("unit", "churn", "variant", "디코딩A", "디코딩B",
             "걸음한계A", "걸음한계B"))
    for unit in units:
        for p in ps:
            for nm in names:
                dpm = pair_means("stuck_decode", unit, p, nm)
                spm = pair_means("step_limit", unit, p, nm)
                if dpm is None or spm is None:
                    continue
                print("   %-15s %-6.2f %-18s %8.1f%% %8.1f%% %8.1f%%"
                      " %8.1f%%"
                      % (unit, p, nm, dpm[0], dpm[1],
                         spm[0], spm[1]))
    print("   걸음한계B가 0이 아니면 규칙을 푼 대가를 치른 것이다.")

    print()
    print("   홉(세 방법이 모두 도달한 에피소드만): 본 측정 -> %s" % label)
    print("   %-15s %-6s %-18s %8s %8s %8s"
          % ("unit", "churn", "variant", "본측정", label, "차"))
    for unit in units:
        for p in ps:
            for nm in names:
                hpm = pair_means("paired_hop", unit, p, nm)
                if hpm is None:
                    continue
                print("   %-15s %-6.2f %-18s %8.2f %8.2f %+8.2f  (셀 %d)"
                      % (unit, p, nm, hpm[0], hpm[1],
                         hpm[1] - hpm[0], hpm[2]))
    print("   홉은 걸음 수이고 규칙을 풀면 두 통로가 반대로 움직인다 —")
    print("   플래너가 쓸 그래프가 넓어져 짧아지는 쪽과 되밟아 늘어나는 쪽.")
    print("   CSV만으로는 못 가르므로 --decode-probe의 노드 열과 함께 읽는다.")

    print()
    print("   P1 재검정 — %s 규칙에서 제안 빼기 기준선 도달률 차" % label)
    q1 = []
    for unit in units:
        for p in sorted({r["churn_p"] for r in rb}):
            if p == 0.0 and unit != "link":
                continue
            for base in names[1:]:
                d = paired_diff(rb, "success",
                                (unit, p, "risk_dqn"), (unit, p, base))
                if len(d) < 3:
                    continue
                m, mde, pv = paired_t(d)
                q1.append((unit, p, base, m, mde, pv))
    print("   %-15s %-6s %-18s %7s %6s %8s %8s"
          % ("unit", "churn", "기준선", "차(%p)", "MDE", "p", "보정p"))
    alive = []
    for x, aq in zip(q1, holm_bonferroni([z[5] for z in q1])):
        hit = aq == aq and aq < 0.05
        if hit:
            alive.append(x)
        print("   %-15s %-6.2f %-18s %+7.2f %6.2f %8s %8s%s"
              % (x[0], x[1], x[2], x[3], x[4], fmt_p(x[5]), fmt_p(aq),
                 "" if not hit else " *"))
    print("   비교 %d번 Holm 보정 후 살아남은 차 %d건."
          % (len(q1), len(alive)))
    print("   본 측정에서는 2건이었다(node_protected 0.50, 보정 p=0.0032")
    print("   / 0.0099). 0건이면 그 결손은 라우팅이 아니라 규칙이 만든 것이다.")

    print()
    print("   P3 재검정 — 서 있는 노드를 보호하면 도달률이 100이 되는가")
    print("   %-6s %-18s %8s %8s %8s %8s"
          % ("churn", "variant", "도달", "고립", "디코딩", "걸음한계"))
    for p in ps:
        for nm in names:
            vb = per_seed(rb, "node_protected", p, nm, "success")
            if not vb:
                continue
            gi = per_seed(rb, "node_protected", p, nm, "stuck_isolated")
            gd = per_seed(rb, "node_protected", p, nm, "stuck_decode")
            gs = per_seed(rb, "node_protected", p, nm, "step_limit")
            print("   %-6.2f %-18s %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
                  % (p, nm, sum(vb) / len(vb), sum(gi) / len(gi),
                     sum(gd) / len(gd), sum(gs) / len(gs)))


def report():
    rows = load_rows()
    units = [u for u in CHURN_UNITS
             if any(r["churn_unit"] == u for r in rows)]
    ps = sorted({r["churn_p"] for r in rows})
    names = [v for v, _ in VARIANTS]

    print("=" * 74)
    print("C-1 축소형 — 에피소드 중간 노드/링크 이탈")
    eps = sorted({int(r["episodes"]) for r in rows if "episodes" in r})
    print("시나리오 %d × 시드 %d, 에피소드 %s개/조건"
          % (len({r["scenario"] for r in rows}),
             len({r["seed"] for r in rows}),
             "/".join(str(e) for e in eps) or "?"))
    print("=" * 74)

    cells = {}
    for r in rows:
        key = (r["scenario"], r["churn_unit"], r["churn_p"], r["seed"])
        cells.setdefault(key, []).append(r["paired_n"])
    bad = [k for k, v in cells.items() if len(set(v)) > 1]
    print()
    print("0) 정합성 — 한 셀의 세 방법이 같은 짝 집합을 쓰는가")
    print("   셀 %d개 중 짝 n이 어긋난 셀 %d개%s"
          % (len(cells), len(bad),
             "" if not bad else "  ← 짝 채점이 비대칭이다, 수치 쓰지 말 것"))
    for k in bad[:3]:
        print("     %s %s churn=%.2f seed=%d → %s"
              % (k[0], k[1], k[2], k[3], cells[k]))

    print("\n1) 도달률 % (시나리오·시드 평균, ± 는 시드 간 표준편차)")
    print("   %-16s %-6s %s" % ("unit", "churn",
                                "".join("%18s" % n for n in names)))
    for unit in units:
        for p in ps:
            if p == 0.0 and unit != "link":
                continue
            cells = ""
            for n in names:
                vals = per_seed(rows, unit, p, n, "success")
                if not vals:
                    cells += "%18s" % "-"
                    continue
                m, s = mean_std(vals)
                cells += "%18s" % ("%.1f ± %.1f" % (m, s))
            print("   %-16s %-6.2f %s" % (unit, p, cells))

    print()
    print("1-1) 표 1의 짝 검정 — 겹치는 ±를 눈으로 재지 않는다")
    print("   P1: 같은 (시나리오, 시드)에서 제안 − 기준선 도달률 차")
    q1 = []
    for unit in units:
        for p in ps:
            if p == 0.0 and unit != "link":
                continue
            for base in names[1:]:
                d = paired_diff(rows, "success",
                                (unit, p, "risk_dqn"), (unit, p, base))
                if len(d) < 3:
                    continue
                m, mde, pv = paired_t(d)
                q1.append((unit, p, base, m, mde, pv))
    a1 = holm_bonferroni([x[5] for x in q1])
    print("   %-15s %-6s %-17s %7s %6s %8s %8s"
          % ("unit", "churn", "기준선", "차(%p)", "MDE", "p", "보정p"))
    for x, aq in zip(q1, a1):
        print("   %-15s %-6.2f %-17s %+7.2f %6.2f %8s %8s%s"
              % (x[0], x[1], x[2], x[3], x[4], fmt_p(x[5]), fmt_p(aq),
                 "" if not (aq == aq and aq < 0.05) else " *"))
    print("   비교 %d번이므로 Holm 보정한 열로만 읽는다(* = 보정 후 유의)."
          % len(q1))
    print("   짝 n = 시나리오 4 × 시드 5 = 20. |차| < MDE는 '차이 없음'이")
    print("   아니라 '이 표본으로는 가릴 수 없다'로 읽는다.")

    print()
    print("   P6: 같은 (시나리오, 시드, 방법)에서 link − node 도달률 차")
    q6 = []
    for p in [x for x in ps if x > 0]:
        for nm in names:
            d = paired_diff(rows, "success",
                            ("link", p, nm), ("node", p, nm))
            if len(d) < 3:
                continue
            m, mde, pv = paired_t(d)
            q6.append((p, nm, m, mde, pv, len(d)))
    for x, aq in zip(q6, holm_bonferroni([z[4] for z in q6])):
        print("   churn=%.2f %-18s %+7.2f%%p  MDE %.2f  보정p=%s  (짝 %d)"
              % (x[0], x[1], x[2], x[3], fmt_p(aq), x[5]))

    print("\n2) 실패 원인 분해 % (자기 노드 고립 / 경로 없음 / 디코딩 규칙)")
    print("   %-16s %-6s %-18s %8s %8s %8s"
          % ("unit", "churn", "variant", "고립", "경로없음", "디코딩"))
    for unit in units:
        for p in ps:
            if p == 0.0:
                continue
            for n in names:
                v = [mean_std(per_seed(rows, unit, p, n, f))[0]
                     if per_seed(rows, unit, p, n, f) else float("nan")
                     for f in ("stuck_isolated", "stuck_nopath",
                               "stuck_decode")]
                print("   %-16s %-6.2f %-18s %8.1f %8.1f %8.1f"
                      % (unit, p, n, v[0], v[1], v[2]))

    print("\n3) 반증 — 서 있는 노드를 보호하면 실패가 사라지는가")
    for p in [x for x in ps if x > 0]:
        for n in names:
            a = per_seed(rows, "node", p, n, "success")
            b = per_seed(rows, "node_protected", p, n, "success")
            if not a or not b:
                continue
            diff, _, _, pv = welch(b, a)
            print("   churn=%.2f %-18s node %.1f → 보호 %.1f"
                  "  (차 %+.1f%%p, p=%s)"
                  % (p, n, mean_std(a)[0], mean_std(b)[0], diff, fmt_p(pv)))

    print("\n4) 품질 — 둘 다 도달한 에피소드만(짝 채점)")
    print("   세 방법이 모두 도달한 에피소드만 쓴다 — 제안 방법도 예외가 아니다.")
    print("   제안 방법 대 공정 비교군(ospf_delay_replan). 양수면 제안이 낫다.")
    q4 = []
    for metric, label in (("compounded", "복합 손실확률"),
                          ("delay", "종단간 지연"),
                          ("hop", "홉 수")):
        print("   [%s]" % label)
        for unit in units:
            for p in ps:
                if p == 0.0 and unit != "link":
                    continue
                a = per_seed(rows, unit, p, "risk_dqn", "paired_" + metric)
                b = per_seed(rows, unit, p, "ospf_delay_replan",
                             "paired_" + metric)
                if len(a) < 2 or len(b) < 2:
                    continue
                diff, _, _, pv = welch(b, a)
                mde = min_detectable_effect(b, a)
                base = mean_std(b)[0]
                rel = diff / base * 100.0 if base else 0.0
                nmin = min(per_seed(rows, unit, p, "ospf_delay_replan",
                                    "paired_n") or [0])
                q4.append((label, unit, p, diff, pv, mde))
                print("     %-16s churn=%.2f  %+.4f (%+.1f%%)  p=%s"
                      "  MDE %.4f  짝n>=%d"
                      % (unit, p, diff, rel, fmt_p(pv), mde, nmin))

    print()
    print("4-1) 다중비교 보정 — 4절은 한 번이 아니라 %d번의 비교다" % len(q4))
    adj = holm_bonferroni([x[4] for x in q4])
    raw = [x for x in q4 if x[4] == x[4] and x[4] < 0.05]
    kept = [a for a in adj if a == a and a < 0.05]
    print("   보정 전 p<0.05 인 비교 %d개 → Holm 보정 후 %d개"
          % (len(raw), len(kept)))
    pairs = sorted(zip(q4, adj), key=lambda z: (z[0][4] != z[0][4], z[0][4]))
    for x, aq in pairs[:4]:
        print("     %-12s %-16s churn=%.2f  %+.4f  p=%s → 보정 %s%s"
              % (x[0], x[1], x[2], x[3], fmt_p(x[4]), fmt_p(aq),
                 "" if not (aq == aq and aq < 0.05) else "  ←보정 후 유의"))
    print("   3지표 × 7조건 = %d번이므로 보정 없는 p=0.04 하나는 우연으로"
          " 기대되는 수다." % len(q4))

    print()
    print("4-2) 짝 채점이 결론을 바꿨는가 — 홉, 비짝 대 짝")
    print("   %-15s %-6s %10s %10s" % ("unit", "churn", "비짝", "짝"))
    flips, gap = 0, 0.0
    for unit in units:
        for p in ps:
            if p == 0.0 and unit != "link":
                continue
            out = []
            for field in ("hop", "paired_hop"):
                a = per_seed(rows, unit, p, "risk_dqn", field)
                b = per_seed(rows, unit, p, "ospf_delay_replan", field)
                out.append(welch(b, a)[0]
                           if len(a) >= 2 and len(b) >= 2 else float("nan"))
            if out[0] != out[0] or out[1] != out[1]:
                continue
            flips += 1 if out[0] * out[1] < 0 else 0
            gap = max(gap, abs(out[0] - out[1]))
            print("   %-15s %-6.2f %+10.4f %+10.4f %s"
                  % (unit, p, out[0], out[1],
                     "← 부호 뒤집힘" if out[0] * out[1] < 0 else ""))
    print("   비짝은 실패한 에피소드가 빠져 짧은 경로만 남는 생존자 편향을")
    print("   탄다. 뒤집힌 줄 %d개, 비짝−짝 최대 차 %.4f홉." % (flips, gap))
    print("   뒤집힌 줄이 0이면 홉 비교는 생존자 편향이 아니라 실제 경로")
    print("   길이 차를 재고 있다 — 공통 마스크가 결론을 바꾸지 않았다는 뜻.")

    print()
    print("5) 노출 공변량 — 같은 고장 조건 안에서 홉이 긴 방법이 더 실패하는가")
    xs, ys, groups = [], [], 0
    for sc in sorted({r["scenario"] for r in rows}):
        for unit in units:
            for p in ps:
                if p == 0.0:
                    continue
                hs, fs = [], []
                for n in names:
                    h = per_seed(rows, unit, p, n, "hop", sc)
                    v = per_seed(rows, unit, p, n, "success", sc)
                    if h and v:
                        hs.append(mean_std(h)[0])
                        fs.append(100.0 - mean_std(v)[0])
                if len(hs) < len(names):
                    continue
                mh = sum(hs) / len(hs)
                mf = sum(fs) / len(fs)
                xs += [h - mh for h in hs]
                ys += [f - mf for f in fs]
                groups += 1
    r5 = pearson(xs, ys)
    df5 = len(xs) - groups
    t5 = (r5 * math.sqrt(df5 / (1.0 - r5 * r5))
          if df5 > 0 and r5 == r5 and abs(r5) < 1.0 else float("nan"))
    print("   조건군 %d개(시나리오×단위×확률) 안에서 중심화 후 r(홉, 실패율)"
          " = %+.3f  (점 %d개)" % (groups, r5, len(xs)))
    print("   p=%s  (자유도 %d = 점 %d개 − 군 %d개. 군마다 평균을 뺐으므로"
          " 자유도가 군 수만큼 준다)"
          % (fmt_p(t_two_sided_p(t5, df5)), df5, len(xs), groups))
    print("   조건군 간 차이(고장 확률 자체)를 빼야 노출 가설만 남는다 —"
          " 빼지 않으면")
    print("   확률이 분산을 지배해 홉 효과가 묻힌다(합성 검증에서 0.127로 희석됨).")

    print("\n6) 거부된 고장 — 연결 보장 장치가 얼마나 개입했는가")
    for unit in units:
        for p in ps:
            if p == 0.0:
                continue
            v = per_seed(rows, unit, p, "risk_dqn", "rejected_per_ep")
            d = per_seed(rows, unit, p, "risk_dqn", "mean_down_links")
            if v:
                print("   %-16s churn=%.2f  거부 %.3f회/에피소드,"
                      " 평균 단절 링크 %.2f개"
                      % (unit, p, mean_std(v)[0], mean_std(d)[0]))

    print("\n" + "=" * 74)
    print("판정")
    print("=" * 74)
    iso = [mean_std(per_seed(rows, "node", p, n, "stuck_isolated"))[0]
           for p in ps if p > 0 for n in names
           if per_seed(rows, "node", p, n, "stuck_isolated")]
    nop = [mean_std(per_seed(rows, "node", p, n, "stuck_nopath"))[0]
           for p in ps if p > 0 for n in names
           if per_seed(rows, "node", p, n, "stuck_nopath")]
    dec = [mean_std(per_seed(rows, "node", p, n, "stuck_decode"))[0]
           for p in ps if p > 0 for n in names
           if per_seed(rows, "node", p, n, "stuck_decode")]
    print("- 실패 원인: 고립 평균 %.1f%%, 경로 없음 %.1f%%, 디코딩 %.1f%%"
          % (sum(iso) / max(len(iso), 1), sum(nop) / max(len(nop), 1),
             sum(dec) / max(len(dec), 1)))
    print("- 경로 없음이 0에 가까운 것은 발견이 아니라 **연결 보장 장치의 결과**다")
    print("  (6절의 거부 횟수가 그 개입량이다). 정책이 도달성에 기여할 수 있는")
    print("  유일한 영역을 설계상 배제했다는 뜻이므로 같이 밝혀야 한다.")
    print("- 디코딩 규칙이 만든 실패가 작다면 재방문 배제는 교란이 아니다.")
    print("- 3절에서 보호 조건이 실패를 없애면, 이 실험은 라우팅 품질이 아니라")
    print("  **노출 시간**을 재고 있다(5절의 상관이 그 방향을 확인한다).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--ablate-mark", action="store_true")
    ap.add_argument("--risk-dist", action="store_true")
    ap.add_argument("--churn-check", action="store_true")
    ap.add_argument("--ablate-level", action="store_true")
    ap.add_argument("--mask-work", action="store_true")
    ap.add_argument("--path-variety", action="store_true")
    ap.add_argument("--decode-probe", action="store_true")
    ap.add_argument("--probe-churn", type=float,
                    default=0.50)
    ap.add_argument("--probe-unit", choices=CHURN_UNITS,
                    default="node_protected")
    ap.add_argument("--probe-agree", action="store_true",
                    help="P1 재검정: 같은 고장을 본 셀만으로 다시 검정.")
    ap.add_argument("--probe-cross", action="store_true",
                    help="방법 간 짝 전제 점검: 셋이 같은 고장을 보는가.")
    ap.add_argument("--probe-sync", action="store_true",
                    help="6-2절 전제 점검: 규칙별 실행이 같은 고장을 보는가.")
    ap.add_argument("--compare-rules", action="store_true",
                    help="6-2절: 두 규칙의 CSV를 대조한다.")
    ap.add_argument("--decode-rule", choices=("all", "prev", "none"),
                    default="all",
                    help="평가 래퍼의 재방문 규칙. all이 본 측정값이다.")
    args = ap.parse_args()

    if args.decode_rule != "all":
        # 원래 CSV를 덮지 않는다 — 규칙별로 파일을 가른다.
        global DECODE_RULE, OUT_CSV
        DECODE_RULE = args.decode_rule
        OUT_CSV = OUT_CSV.replace(".csv", "_%s.csv" % args.decode_rule)
        print("디코딩 규칙 = %s · 출력 = %s"
              % (DECODE_RULE, os.path.basename(OUT_CSV)))

    if args.compare_rules:
        compare_rules()
        return
    if args.report_only:
        report()
        return
    if args.probe_agree:
        probe_agree_subset(max(args.episodes // 2, 10),
                           args.probe_churn, args.probe_unit)
        return
    if args.probe_cross:
        probe_churn_cross(max(args.episodes // 12, 5), args.probe_churn)
        return
    if args.probe_sync:
        probe_churn_sync(max(args.episodes // 12, 5), args.probe_churn)
        return
    if args.decode_probe:
        probe_decode(max(args.episodes // 4, 10),
                     args.probe_churn, args.probe_unit)
        return
    if args.path_variety:
        measure_path_variety(max(args.episodes // 4, 10))
        return
    if args.mask_work:
        measure_mask_work(max(args.episodes // 4, 10))
        return
    if args.ablate_level:
        ablate_level(max(args.episodes // 4, 10))
        return
    if args.churn_check:
        verify_churn_independent()
        return
    if args.risk_dist:
        risk_distribution()
        return
    if args.ablate_mark:
        verify_visible()
        ablate_mark(max(args.episodes // 4, 10))
        return
    verify_visible()
    verify()
    if not args.verify_only:
        measure(args.episodes)
        report()


if __name__ == "__main__":
    main()
