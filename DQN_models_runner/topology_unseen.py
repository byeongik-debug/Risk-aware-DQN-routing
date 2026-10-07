# topology_unseen.py
# Unseen topology for generalization test
# 기존 학습 토폴로지와 노드 수는 동일하게 유지하고,
# 링크 구성만 일부 변경한 테스트 전용 토폴로지

NUM_NODES = 25
SOURCE = 0
DESTINATION = 24


EDGE_TYPES = {
    # 1. Core / Backbone
    (0, 1): "backbone",
    (1, 2): "backbone",
    (2, 3): "backbone",
    (0, 4): "backbone",
    (1, 5): "backbone",
    (2, 6): "backbone",
    (3, 7): "backbone",

    # 2. Upper Aggregation
    (4, 5): "normal",
    (5, 6): "normal",
    (6, 7): "normal",
    (4, 8): "normal",
    (5, 9): "normal",      # 변경: 기존 (4, 9) 제거 후 대체
    (5, 10): "normal",
    (6, 11): "normal",
    (7, 12): "normal",

    # 3. Middle Mesh
    (8, 9): "normal",
    (9, 10): "normal",
    (10, 11): "normal",
    (11, 12): "normal",
    (8, 13): "normal",
    (9, 14): "normal",
    (10, 15): "normal",
    (11, 16): "normal",
    (12, 17): "normal",

    # 4. Lower Mesh
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

    # 5. Bottom / Destination Area
    (19, 20): "normal",
    (20, 21): "normal",
    (21, 22): "normal",
    (20, 23): "normal",    # 변경: 기존 backup 성격 경로를 일반 경로로 이동
    (22, 23): "normal",
    (23, 24): "normal",

    # 6. Unseen Additional Links
    # 기존 학습 토폴로지에는 없던 우회 링크 추가
    (4, 9): "backup",      # 기존 normal → backup으로 변경
    (6, 10): "backup",
    (8, 14): "backup",
    (9, 15): "backup",
    (11, 17): "backup",
    (14, 19): "backup",
    (15, 21): "backup",
    (18, 23): "backup",
    (22, 24): "backup",
}


EDGES = list(EDGE_TYPES.keys())


BACKBONE_LINKS = [
    edge for edge, link_type in EDGE_TYPES.items()
    if link_type == "backbone"
]


BACKUP_LINKS = [
    edge for edge, link_type in EDGE_TYPES.items()
    if link_type == "backup"
]