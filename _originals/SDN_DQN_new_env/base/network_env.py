# network_env.py
# SDN 네트워크 환경 시뮬레이터
#
# 25개 노드(Node 0 → Node 24)로 구성된 네트워크에서
# 에이전트가 패킷 경로를 결정하는 강화학습 환경(RL Environment)
#
# 4가지 시나리오:
#   normal     - 일반적인 네트워크 상태
#   congestion - 특정 링크에 트래픽 집중
#   unstable   - 링크 품질이 불규칙하게 변동
#   dynamic    - 혼잡+불안정이 에피소드마다 다른 경로에 발생

import random           # 랜덤 노이즈, 시나리오 선택에 사용
from collections import deque  # 고정 크기 큐 (위험도 히스토리 저장용)

import networkx as nx   # 그래프 자료구조 및 최단 경로 알고리즘
import numpy as np      # 수치 연산, 배열 처리

from topology import NUM_NODES, SOURCE, DESTINATION, EDGES, EDGE_TYPES  # 네트워크 구조 정의
from risk_calculator import calculate_link_risk                           # 링크 위험도 계산 함수

RISK_WINDOW = 5              # 시간적 위험도 계산에 사용할 과거 스텝 수 (k=5)
MAX_STEPS = 35               # 에피소드당 최대 이동 허용 횟수 (초과 시 실패 처리)

DYNAMIC_PHASE_PER_EPISODE = True  # True: 에피소드마다 phase 무작위 선택, False: 스텝 수로 전환

# 혼잡 시나리오에서 트래픽이 집중될 링크 목록
CONGESTION_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]
# 불안정 시나리오에서 품질이 불규칙하게 변할 링크 목록
UNSTABLE_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]

def path_to_edges(path):
    # 노드 번호 리스트를 (u, v) 엣지 튜플 리스트로 변환
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]  # 인접 노드 쌍으로 분리


# dynamic 시나리오에서 phase별로 혼잡/안정 경로를 다르게 설정
PATH_A = path_to_edges([0, 1, 5, 4, 8, 13, 19, 20, 21, 22, 24])  # 우회 경로 (홉 수 많음)
PATH_B = path_to_edges([0, 4, 9, 14, 20, 23, 24])                 # 백본 직통 경로
PATH_C = path_to_edges([0, 1, 5, 10, 15, 20, 23, 24])             # 중간 경유 경로
PATH_D = path_to_edges([0, 1, 5, 11, 17, 22, 24])                 # 하단 우회 경로

# phase 0/1/2마다 위험(danger)·안전(support) 경로가 순환 — 에이전트가 적응해야 함
DYNAMIC_PHASES = {
    0: {
        "danger": PATH_B + PATH_C + PATH_D,  # B·C·D가 위험 → A로 우회 유도
        "support": PATH_A,
    },
    1: {
        "danger": PATH_A + PATH_C,           # A·C가 위험 → B로 우회 유도
        "support": PATH_B,
    },
    2: {
        "danger": PATH_A + PATH_B,           # A·B가 위험 → C로 우회 유도
        "support": PATH_C,
    },
}


class NetworkEnv:
    def __init__(self, use_risk=True, scenario="normal", risk_window=RISK_WINDOW):
        # use_risk: True면 위험도 벡터를 상태로 사용(Risk-DQN), False면 원시 QoS 사용
        # scenario: 시뮬레이션할 네트워크 상황, risk_window: 시간적 위험도 계산에 쓸 과거 스텝 수
        self.use_risk     = use_risk      # Risk-DQN / NoRisk 모드 선택
        self.scenario     = scenario      # 현재 시나리오 (normal/congestion/unstable/dynamic)
        self.risk_window  = risk_window   # 히스토리 큐 크기
        self.timestep     = 0             # 현재 에피소드 내 스텝 카운터
        self.dynamic_phase = 0            # dynamic 시나리오의 현재 phase (0/1/2)

        self.graph = nx.Graph()                       # 네트워크 그래프 생성
        self.graph.add_nodes_from(range(NUM_NODES))   # 25개 노드 추가
        self.graph.add_edges_from(EDGES)              # 링크(엣지) 추가

        self.qos           = {}   # 링크별 QoS 정보 딕셔너리 (지연/손실/대역폭/위험도)
        self.loss_history  = {}   # 링크별 손실률 히스토리 (시간적 위험도 계산용)
        self.delay_history = {}   # 링크별 지연 히스토리 (시간적 위험도 계산용)

        self.current_node = SOURCE  # 에이전트 현재 위치 (시작: Source 노드)
        self.visited      = []      # 이미 방문한 노드 목록 (루프 방지용)

        if self.use_risk:
            self.state_size = len(EDGES) + NUM_NODES * 3       # 위험도 벡터 + 원핫 3개
        else:
            self.state_size = len(EDGES) * 2 + NUM_NODES * 3  # 지연+손실 per edge + 원핫 3개
        self.action_size = NUM_NODES                            # 액션 공간 = 전체 노드 수

        for edge in EDGES:                                      # 모든 링크에 대해 히스토리 큐 초기화
            e = self.normalize_edge(*edge)
            self.loss_history[e]  = deque(maxlen=self.risk_window)  # 손실 히스토리 큐
            self.delay_history[e] = deque(maxlen=self.risk_window)  # 지연 히스토리 큐

        self.init_qos()  # 초기 QoS 상태 생성

    def normalize_edge(self, u, v):
        # 양방향 링크를 항상 (작은번호, 큰번호) 형태로 통일해 딕셔너리 키로 사용
        return tuple(sorted((u, v)))  # 방향 무관하게 동일한 키 보장

    def get_link_type(self, edge):
        # 엣지의 링크 타입(backbone / normal / backup) 반환
        e = self.normalize_edge(*edge)          # 정규화된 엣지 키로 변환
        return EDGE_TYPES.get(
            e,
            EDGE_TYPES.get((edge[1], edge[0]), "normal")  # 없으면 역방향으로 재탐색, 기본값 normal
        )

    def init_qos(self):
        # 에피소드 시작 시 모든 링크의 초기 QoS 상태(지연/손실/대역폭/위험도) 생성
        self.qos = {}  # 기존 QoS 초기화

        for u, v in EDGES:                          # 모든 링크 순회
            e         = self.normalize_edge(u, v)   # 정규화된 링크 키
            link_type = self.get_link_type(e)       # 링크 타입 조회

            if link_type == "backbone":             # 백본: 낮은 지연, 높은 대역폭
                delay     = random.uniform(5, 10)
                loss      = random.uniform(0.003, 0.015)
                bandwidth = random.uniform(800, 1000)

            elif link_type == "backup":             # 우회: 중간 지연, 중간 대역폭
                delay     = random.uniform(15, 25)
                loss      = random.uniform(0.001, 0.010)
                bandwidth = random.uniform(400, 700)

            else:                                   # 일반: 기본 설정
                delay     = random.uniform(10, 20)
                loss      = random.uniform(0.005, 0.020)
                bandwidth = random.uniform(400, 800)

            capacity    = bandwidth                              # 최대 용량 = 대역폭
            traffic     = random.uniform(0.10, 0.35) * capacity  # 초기 트래픽 (용량의 10~35%)
            utilization = traffic / max(capacity, 1e-6)          # 사용률 계산 (0 나누기 방지)
            is_congested = utilization >= 0.80                   # 80% 이상이면 혼잡 상태

            self.qos[e] = {                          # 링크 QoS 정보 딕셔너리
                "delay":        float(delay),
                "loss":         float(loss),
                "bandwidth":    float(bandwidth),
                "capacity":     float(capacity),
                "traffic":      float(traffic),
                "utilization":  float(utilization),
                "is_congested": bool(is_congested),
                "risk":         0.0,                 # 초기 위험도 (아래에서 계산)
                "type":         link_type,
            }

            self.loss_history[e].clear()             # 히스토리 초기화
            self.delay_history[e].clear()
            self.loss_history[e].append(loss)        # 초기값으로 첫 히스토리 설정
            self.delay_history[e].append(delay)

            self.qos[e]["risk"] = calculate_link_risk(  # 초기 위험도 계산
                utilization=utilization,
                is_congested=is_congested,
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=link_type,
            )

    def get_dynamic_phase_sets(self):
        # 현재 dynamic phase에서 위험 링크 집합(danger)과 안정 링크 집합(support) 반환
        phase_info  = DYNAMIC_PHASES[self.dynamic_phase]                                    # 현재 phase 정보 조회
        danger_set  = {self.normalize_edge(*edge) for edge in phase_info["danger"]}         # 위험 링크 집합
        support_set = {self.normalize_edge(*edge) for edge in phase_info["support"]}        # 안전 링크 집합
        return danger_set, support_set

    def update_dynamic_qos(self):
        # 한 스텝 경과: 시나리오에 따라 링크별 트래픽/지연/손실/위험도를 동적으로 변화
        self.timestep += 1  # 스텝 카운터 증가

        congestion_set = {self.normalize_edge(*x) for x in CONGESTION_LINKS}  # 혼잡 링크 집합
        unstable_set   = {self.normalize_edge(*x) for x in UNSTABLE_LINKS}    # 불안정 링크 집합

        dynamic_danger_set  = set()  # dynamic 시나리오 위험 링크 (초기화)
        dynamic_support_set = set()  # dynamic 시나리오 안전 링크 (초기화)

        if self.scenario == "dynamic":                              # dynamic 시나리오일 때만
            if not DYNAMIC_PHASE_PER_EPISODE:                      # 스텝 기반 phase 전환 모드
                phase_keys = list(DYNAMIC_PHASES.keys())
                self.dynamic_phase = phase_keys[(self.timestep // 8) % len(phase_keys)]  # 8스텝마다 phase 전환
            dynamic_danger_set, dynamic_support_set = self.get_dynamic_phase_sets()       # 현재 phase 집합 조회

        for u, v in EDGES:                          # 모든 링크에 대해 QoS 업데이트
            e         = self.normalize_edge(u, v)   # 정규화된 링크 키
            q         = self.qos[e]                 # 현재 QoS 정보 참조
            link_type = q["type"]                   # 링크 타입
            capacity  = q["capacity"]               # 최대 용량

            # 기본 노이즈 (모든 시나리오 공통)
            delay_noise   = random.uniform(-1.0, 1.0)    # 지연 변동 노이즈
            loss_noise    = random.uniform(-0.002, 0.002) # 손실 변동 노이즈
            bw_noise      = random.uniform(-5.0, 15.0)   # 대역폭 변동 노이즈
            traffic_ratio = random.uniform(0.15, 0.45)   # 기본 트래픽 비율
            risk_type     = link_type                     # 기본 위험 타입 = 링크 타입

            if self.scenario == "congestion" and e in congestion_set:       # 혼잡 링크 처리
                traffic_ratio = random.uniform(1.10, 1.55)                  # 용량 초과 트래픽
                delay_noise   = random.uniform(5.0, 10.0)                   # 지연 급증
                loss_noise    = random.uniform(0.012, 0.050)                # 손실 급증
                bw_noise      = random.uniform(0.0, 30.0)                   # 대역폭 변동
                risk_type     = "congested"                                  # 위험 타입 변경

            elif self.scenario == "unstable" and e in unstable_set:         # 불안정 링크 처리
                traffic_ratio = random.uniform(0.55, 1.10)                  # 중간~초과 트래픽
                delay_noise   = random.uniform(-5.0, 10.0)                  # 지연 불규칙 변동
                loss_noise    = random.uniform(-0.010, 0.055)               # 손실 불규칙 변동
                bw_noise      = random.uniform(0.0, 25.0)                   # 대역폭 변동
                risk_type     = "unstable"                                   # 위험 타입 변경

            elif self.scenario == "dynamic" and e in dynamic_danger_set:    # dynamic 위험 링크 처리
                traffic_ratio = random.uniform(1.10, 1.55)                  # 혼잡+불안정 조합
                delay_noise   = random.uniform(5.0, 10.0) + random.uniform(-5.0, 10.0)
                loss_noise    = random.uniform(0.012, 0.050) + random.uniform(-0.010, 0.055)
                bw_noise      = random.uniform(0.0, 30.0)
                risk_type     = "unstable_congested"                         # 최고 위험 타입

            if link_type == "backbone":          # 백본 링크는 노이즈 감쇠 (안정적)
                delay_noise   *= 0.6
                loss_noise    *= 0.6
                traffic_ratio *= 0.95

            elif link_type == "backup":          # 우회 링크도 노이즈 일부 감쇠
                delay_noise   *= 0.75
                loss_noise    *= 0.6
                traffic_ratio *= 0.75

            # QoS 값 업데이트 (클리핑으로 범위 제한)
            q["traffic"]     = float(np.clip(traffic_ratio * capacity, 0.0, capacity * 1.5))            # 트래픽
            q["utilization"] = float(np.clip(q["traffic"] / max(capacity, 1e-6), 0.0, 1.5))            # 사용률
            q["delay"]       = float(np.clip(q["delay"] + delay_noise + q["utilization"] * 2.0, 3.0, 120.0))  # 지연 (ms)
            q["loss"]        = float(np.clip(q["loss"] + loss_noise + max(0.0, q["utilization"] - 0.8) * 0.03, 0.0, 0.35))  # 손실률
            q["bandwidth"]   = float(np.clip(q["bandwidth"] + bw_noise, 50.0, 1000.0))                 # 대역폭
            q["is_congested"] = q["utilization"] >= 0.8                                                 # 혼잡 여부 갱신

            self.loss_history[e].append(q["loss"])    # 손실 히스토리 업데이트
            self.delay_history[e].append(q["delay"])  # 지연 히스토리 업데이트

            q["risk"] = calculate_link_risk(          # 위험도 재계산
                utilization=q["utilization"],
                is_congested=q["is_congested"],
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=risk_type,
            )

            if self.scenario in ["congestion", "unstable"] and link_type == "backup":
                q["risk"] = float(np.clip(q["risk"] * 0.60, 0.0, 1.0))  # 우회 링크는 위험도 완화

            if self.scenario == "dynamic" and e in dynamic_support_set:
                q["risk"] = float(np.clip(q["risk"] * 0.55, 0.0, 1.0))  # 안전 경로는 위험도 추가 완화

    def reset(self):
        # 에피소드 시작: 에이전트를 출발 노드로 초기화하고 초기 상태 벡터 반환
        self.current_node = SOURCE       # 에이전트 위치를 출발 노드로 리셋
        self.visited      = [SOURCE]     # 방문 노드 목록 초기화 (출발 노드 포함)
        self.timestep     = 0            # 스텝 카운터 초기화

        if self.scenario == "dynamic":   # dynamic 시나리오: 매 에피소드 phase 무작위 선택
            self.dynamic_phase = random.choice(list(DYNAMIC_PHASES.keys()))
        else:
            self.dynamic_phase = 0       # 다른 시나리오: phase 0 고정

        self.init_qos()           # QoS 초기화
        self.update_dynamic_qos() # 첫 스텝 QoS 업데이트
        return self.get_state()   # 초기 상태 벡터 반환

    def get_valid_actions(self, node=None):
        # 현재(또는 지정) 노드에서 이동 가능한 인접 노드 번호 목록 반환
        target_node = self.current_node if node is None else node  # 노드 미지정 시 현재 위치 사용
        return list(self.graph.neighbors(target_node))              # 그래프 인접 노드 목록 반환

    def shortest_distance_to_destination(self, node):
        # 특정 노드에서 목적지까지 최단 홉 수 반환 (보상 계산 시 방향성 판단에 사용)
        try:
            return nx.shortest_path_length(self.graph, node, DESTINATION)  # NetworkX 최단 경로 함수 사용
        except nx.NetworkXNoPath:
            return 999  # 경로 없음 → 매우 큰 값 반환 (실질적 무한대)

    def get_state(self):
        # 현재 네트워크 상태를 에이전트 입력 벡터로 변환
        # [링크별 위험도] + [현재 노드 원핫] + [목적지 원핫] + [방문 노드 원핫]
        state = []
        if self.use_risk:                          # Risk-DQN: 링크별 위험도 사용
            for u, v in EDGES:
                e = self.normalize_edge(u, v)
                state.append(self.qos[e]["risk"])  # 각 링크의 위험도 (0~1)
        else:                                      # NoRisk: 지연/손실 원시값 사용
            for u, v in EDGES:
                e = self.normalize_edge(u, v)
                q = self.qos[e]
                state.append(float(np.clip(q["delay"] / 120.0, 0.0, 1.0)))  # 지연 정규화 (max 120ms)
                state.append(float(np.clip(q["loss"]  / 0.35,  0.0, 1.0)))  # 손실률 정규화 (max 0.35)

        curr_onehot = np.zeros(NUM_NODES)           # 현재 노드 원핫 벡터
        curr_onehot[self.current_node] = 1.0        # 현재 위치에 1 표시

        dest_onehot = np.zeros(NUM_NODES)           # 목적지 원핫 벡터
        dest_onehot[DESTINATION] = 1.0              # 목적지에 1 표시

        visited_onehot = np.zeros(NUM_NODES)        # 방문 노드 원핫 벡터
        for node in self.visited:
            visited_onehot[node] = 1.0              # 방문한 노드에 1 표시

        return np.concatenate([state, curr_onehot, dest_onehot, visited_onehot]).astype(np.float32)  # 벡터 이어붙여 반환

    def step(self, action):
        # 에이전트가 선택한 노드로 이동, (다음상태, 보상, 종료여부, 정보) 반환
        # 시나리오별로 다른 보상 구조 적용
        self.update_dynamic_qos()          # 스텝 시작 시 QoS 업데이트

        valid_actions = self.get_valid_actions()  # 현재 위치에서 유효한 액션 목록

        if action not in valid_actions:    # 유효하지 않은 액션 선택 시 즉시 에피소드 종료
            return (
                self.get_state(),
                -10.0,                     # 큰 페널티
                True,                      # 종료
                {
                    "reason": "invalid",
                    "path": self.visited.copy(),
                    "dynamic_phase": self.dynamic_phase,
                    "next_valid_mask": np.zeros(NUM_NODES, dtype=np.float32),  # 빈 마스크
                }
            )

        e         = self.normalize_edge(self.current_node, action)  # 현재→다음 링크 키
        q         = self.qos[e]                                      # 해당 링크 QoS 정보

        prev_dist = self.shortest_distance_to_destination(self.current_node)  # 이동 전 목적지까지 거리
        next_dist = self.shortest_distance_to_destination(action)             # 이동 후 목적지까지 거리

        risk   = q["risk"]  # 현재 링크 위험도
        reward = 0.0        # 보상 초기화

        if not self.use_risk:                                    # NoRisk 모드: 지연/손실 기반 보상
            reward += 1.5 if next_dist < prev_dist else -1.5    # 목적지 방향 이동 보상
            reward -= q["delay"] * 0.015                         # 지연 패널티
            reward -= q["loss"]  * 8.0                           # 손실 패널티
            if q.get("is_congested", False):
                reward -= 2.0                                    # 혼잡 링크 추가 패널티
            reward -= 0.08                                       # 스텝 페널티 (홉 수 최소화 유도)

        elif self.scenario == "normal":                          # Normal: 위험도 중간 반영
            reward -= risk * 0.7
            reward += 2.0 if next_dist < prev_dist else -2.0
            reward -= 0.10

        elif self.scenario == "congestion":                      # Congestion: 위험도 강하게 반영
            reward -= risk * 10.0
            if q.get("is_congested", False):
                reward -= 4.0                                    # 혼잡 링크 사용 시 추가 패널티
            reward += 0.7 if next_dist < prev_dist else -0.7
            reward -= 0.06

        elif self.scenario == "unstable":                        # Unstable: 높은 위험도 구간 회피 유도
            reward -= risk * 10.0
            if risk >= 0.22:
                reward -= 3.0                                    # 위험 임계값 초과 시 추가 패널티
            reward += 0.7 if next_dist < prev_dist else -0.7
            reward -= 0.06

        elif self.scenario == "dynamic":                         # Dynamic: 위험도 가장 강하게 반영
            reward -= risk * 22.0
            if q.get("is_congested", False):
                reward -= 5.0
            if risk >= 0.25:
                reward -= 5.0
            reward += 0.35 if next_dist < prev_dist else -0.35
            reward -= 0.08

        else:                                                    # 기타 시나리오 기본값
            reward -= risk * 5.0
            reward += 1.0 if next_dist < prev_dist else -1.0
            reward -= 0.05

        if action in self.visited:     # 이미 방문한 노드로 이동 시 루프 패널티
            reward -= 8.0

        self.current_node = action     # 에이전트 위치 업데이트
        self.visited.append(action)    # 방문 노드 목록에 추가

        done = False  # 종료 여부 초기화

        if self.current_node == DESTINATION:                        # 목적지 도달 → 성공 종료
            path_metrics = self.calculate_path_metrics(self.visited) # 경로 성능 지표 계산
            reward += 45.0                                           # 도달 보상
            if self.use_risk:
                reward -= path_metrics["total_risk"] * 6.0          # 경로 전체 위험도 패널티
            done = True

        elif len(self.visited) >= MAX_STEPS:   # 최대 스텝 초과 → 실패 종료
            reward -= 15.0                     # 타임아웃 패널티
            done = True

        next_valid      = self.get_valid_actions(self.current_node)          # 다음 상태의 유효 액션 목록
        next_valid_mask = np.zeros(NUM_NODES, dtype=np.float32)              # 유효 액션 마스크 초기화
        for act in next_valid:
            next_valid_mask[act] = 1.0                                       # 유효 액션 위치에 1 표시

        return (
            self.get_state(),   # 다음 상태 벡터
            reward,             # 보상
            done,               # 종료 여부
            {                   # 추가 정보 딕셔너리
                "reason":        "arrived" if self.current_node == DESTINATION else "continue",
                "path":          self.visited.copy(),
                "dynamic_phase": self.dynamic_phase,
                "delay":         q["delay"],
                "loss":          q["loss"],
                "bandwidth":     q["bandwidth"],
                "risk":          q["risk"],
                "link_type":     q["type"],
                "traffic":       q["traffic"],
                "utilization":   q["utilization"],
                "is_congested":  q["is_congested"],
                "next_valid_mask": next_valid_mask,
            }
        )

    def calculate_path_metrics(self, path):
        # 경로의 총 지연·손실·위험도·홉 수 등 성능 지표 계산 (평가용)
        total_delay    = 0.0           # 경로 전체 지연 합계
        total_loss     = 0.0           # 경로 전체 손실 합계
        total_risk     = 0.0           # 경로 전체 위험도 합계
        min_bandwidth  = float("inf")  # 경로 병목 대역폭 (최솟값)
        unstable_count = 0             # 불안정 링크 수
        congested_count = 0            # 혼잡 링크 수

        unstable_set   = {self.normalize_edge(*x) for x in UNSTABLE_LINKS}    # 불안정 링크 집합
        congestion_set = {self.normalize_edge(*x) for x in CONGESTION_LINKS}  # 혼잡 링크 집합

        dynamic_danger_set = set()  # dynamic 시나리오 위험 링크 집합
        if self.scenario == "dynamic":
            dynamic_danger_set, _ = self.get_dynamic_phase_sets()  # 현재 phase 위험 집합 조회

        hop_count = max(len(path) - 1, 1)  # 홉 수 (평균 계산 시 0 나누기 방지)

        for u, v in zip(path[:-1], path[1:]):   # 경로의 각 링크 순회
            e = self.normalize_edge(u, v)
            if e not in self.qos:               # QoS 정보 없는 링크 skip
                continue
            q = self.qos[e]

            total_delay   += q["delay"]         # 지연 누적
            total_loss    += q["loss"]           # 손실 누적
            total_risk    += q["risk"]           # 위험도 누적
            min_bandwidth  = min(min_bandwidth, q["bandwidth"])  # 병목 대역폭 갱신

            if self.scenario == "unstable" and e in unstable_set:       # 불안정 링크 카운트
                unstable_count += 1
            elif self.scenario == "congestion" and e in congestion_set:  # 혼잡 링크 카운트
                congested_count += 1
            elif self.scenario == "dynamic" and e in dynamic_danger_set: # dynamic 위험 링크
                unstable_count  += 1
                congested_count += 1

        if min_bandwidth == float("inf"):  # 링크 정보 없으면 0으로 처리
            min_bandwidth = 0.0

        return {                                       # 경로 성능 지표 딕셔너리 반환
            "path":           path,
            "hop_count":      len(path) - 1,           # 총 홉 수
            "total_delay":    total_delay,             # 전체 지연 (ms)
            "total_loss":     total_loss / hop_count,  # 평균 손실률
            "total_risk":     total_risk / hop_count,  # 평균 위험도
            "min_bandwidth":  min_bandwidth,           # 병목 대역폭
            "unstable_count": unstable_count,          # 불안정 링크 수
            "congested_count": congested_count,        # 혼잡 링크 수
            "dynamic_phase":  self.dynamic_phase,      # 현재 phase
        }

    def ospf_path(self, cost_type="bandwidth"):
        # OSPF 방식으로 최단 경로 탐색 — 비교 기준선(baseline)으로 사용
        g = nx.Graph()                  # 비용 가중치 그래프 생성
        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]

            if cost_type == "delay":                             # 지연 기반 비용
                cost = q["delay"]
            elif cost_type == "hop":                             # 홉 수 기반 비용
                cost = 1
            elif cost_type == "bandwidth":                       # 대역폭 기반 비용 (역수)
                cost = 1000.0 / max(q["bandwidth"], 1e-6)
            elif cost_type == "risk":                            # 위험도 기반 비용
                cost = q["risk"]
            else:                                                # 기본값: 지연
                cost = q["delay"]

            g.add_edge(u, v, weight=cost)  # 비용을 가중치로 엣지 추가

        return nx.shortest_path(g, SOURCE, DESTINATION, weight="weight")  # 최저 비용 경로 반환