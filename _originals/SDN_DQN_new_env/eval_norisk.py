# eval_norisk.py
# 학습된 NoRisk-DQN(norisk_model_final.pth)을 5개 시드로 반복 평가
# 결과를 norisk_reference.csv로 저장 → show_results.py에서 실험 4로 표시

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
from network_env import NetworkEnv
from topology import DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "norisk_model_final.pth")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_CSV = os.path.join(RESULT_DIR, "norisk_reference.csv")

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EVAL_EPISODES = 200
SEEDS = [0, 1, 2, 3, 4]   # 5개 시드 → mean ± std


class QNetwork(nn.Module):
    """train_norisk_dqn.py와 동일한 구조 — 저장된 가중치 로드를 위해 일치해야 함"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


def _evaluate_one_seed(model, scenario, device, seed, episodes=EVAL_EPISODES):
    """단일 시드로 NoRisk-DQN 평가 → 이 시드의 평균값 반환"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, hops = [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(state_t)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid_actions:
                masked[a] = q[a]
            action = int(torch.argmax(masked).item())
            state, _, done, _ = env.step(action)

        if env.current_node == DESTINATION:
            success_count += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
            hop_count = max(m["hop_count"], 1)
            unstable_usages.append(m["unstable_count"] / hop_count * 100)
            congested_usages.append(m["congested_count"] / hop_count * 100)

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
        "unstable_usage":  float(np.mean(unstable_usages))  if unstable_usages  else 0.0,
        "congested_usage": float(np.mean(congested_usages)) if congested_usages else 0.0,
    }


def evaluate(model, scenario, device):
    """5개 시드로 반복 평가 → mean ± std (ddof=1) 반환"""
    seed_results = [_evaluate_one_seed(model, scenario, device, s) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "scenario": scenario,
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":      float(np.mean(d_vals)),
        "delay_std":  float(np.std(d_vals, ddof=1)),
        "loss":       float(np.mean(l_vals)),
        "loss_std":   float(np.std(l_vals, ddof=1)),
        "hop":        float(np.mean(h_vals)),
        "hop_std":    float(np.std(h_vals, ddof=1)),
        "unstable_usage":  float(np.mean([r["unstable_usage"]  for r in seed_results])),
        "congested_usage": float(np.mean([r["congested_usage"] for r in seed_results])),
    }


def main():
    print("=" * 70)
    print(f"  NoRisk DQN 평가  (seeds={SEEDS})")
    print(f"  모델: {MODEL_PATH}")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = NetworkEnv(use_risk=False, scenario="normal")
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for sc in SCENARIOS:
        print(f"\n[평가 중] {sc}  (5 seeds × {EVAL_EPISODES} episodes)...")
        r = evaluate(model, sc, device)
        results.append(r)
        print(f"  Delay={r['delay']:.1f} ± {r['delay_std']:.1f} | "
              f"Loss={r['loss']:.5f} ± {r['loss_std']:.5f}")

    fields = ["scenario", "success_rate",
              "delay", "delay_std", "loss", "loss_std",
              "hop", "hop_std", "unstable_usage", "congested_usage"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n저장 완료: {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()