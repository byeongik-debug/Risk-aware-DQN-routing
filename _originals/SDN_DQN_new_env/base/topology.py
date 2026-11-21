# topology.py
# 25개 노드로 구성된 SDN 네트워크 토폴로지 정의
# 노드 0 (SOURCE) → 노드 24 (DESTINATION) 경로를 찾는 게 목표

NUM_NODES = 25    # 전체 노드 수
SOURCE = 0        # 패킷 출발 노드
DESTINATION = 24  # 패킷 도착 노드

# 각 엣지(링크)의 타입을 정의
# 타입별로 네트워크 환경에서 QoS 특성이 다르게 적용됨
EDGE_TYPES = {
    # 1. Core / Backbone: 주요 고속 링크 (낮은 지연, 높은 대역폭)
    (0, 1): "backbone",
    (1, 2): "backbone",
    (2, 3): "backbone",
    (0, 4): "backbone",
    (1, 5): "backbone",
    (2, 6): "backbone",
    (3, 7): "backbone",

    # 2. Upper Aggregation: 백본에서 메쉬로 연결되는 집선 구간
    (4, 5): "normal",
    (5, 6): "normal",
    (6, 7): "normal",
    (4, 8): "normal",
    (4, 9): "normal",
    (5, 10): "normal",
    (6, 11): "normal",
    (7, 12): "normal",

    # 3. Middle Mesh: 중간 메쉬 구간 (다양한 경로 선택 가능)
    (8, 9): "normal",
    (9, 10): "normal",
    (10, 11): "normal",
    (11, 12): "normal",
    (8, 13): "normal",
    (9, 14): "normal",
    (10, 15): "normal",
    (11, 16): "normal",
    (12, 17): "normal",

    # 4. Lower Mesh: 목적지 방향 하위 메쉬
    (13, 14): "normal",
    (14, 15): "normal",
    (15, 16): "normal",
    (16, 17): "normal",
    (17, 18): "normal",
    (13, 19): "normal",
    (14, 20): "normal",
    (15, 20): "normal",
    (16, 21): "normal",
    (17, 22): "normal",
    (18, 22): "normal",

    # 5. Bottom / Destination Area: 목적지 근처 구간
    (19, 20): "normal",
    (20, 21): "normal",
    (21, 22): "normal",
    (21, 23): "normal",
    (22, 23): "normal",
    (23, 24): "normal",

    # 6. Backup / Bypass Links: 우회 경로용 링크
    # 평상시엔 잘 안 쓰이지만 혼잡 시 대안 경로로 활용
    (2, 5): "backup",
    (5, 11): "backup",
    (7, 11): "backup",
    (8, 14): "backup",
    (11, 17): "backup",
    (14, 19): "backup",
    (15, 21): "backup",
    (20, 23): "backup",
    (22, 24): "backup",
}

# 딕셔너리 키에서 엣지 리스트만 추출 (양방향 아님, 단방향 정의)
EDGES = list(EDGE_TYPES.keys())

# 백본 링크만 따로 추출 (위험도 계산 시 링크 타입 구분에 사용)
BACKBONE_LINKS = [
    edge for edge, link_type in EDGE_TYPES.items()
    if link_type == "backbone"
]

# 우회 링크만 따로 추출
BACKUP_LINKS = [
    edge for edge, link_type in EDGE_TYPES.items()
    if link_type == "backup"
]