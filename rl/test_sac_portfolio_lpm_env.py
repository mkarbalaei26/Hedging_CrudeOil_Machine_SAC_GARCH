import numpy as np
from rl.SACPortfolioLPMEnv import SACPortfolioLPMEnv, SACPortfolioLPMConfig

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTE_PATH = PROJECT_ROOT / "rl_cache" / "precompute_WTI.npz"

pre = np.load(PRECOMPUTE_PATH, allow_pickle=True)
scenarios = [
    {
        "start_idx": 5000,
        "end_idx": 5030,
        "volume_bbl": 1_000_000,
    }
]

cfg = SACPortfolioLPMConfig(info_mode="eval")

env = SACPortfolioLPMEnv(pre, scenarios, cfg=cfg, seed=42)

obs, info = env.reset(options={"scenario_idx": 0})
print("obs shape:", obs.shape)
print("reset info:", info)

done = False
rows = []

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rows.append(info)
    done = terminated or truncated

print("steps:", len(rows))
print("last info:")
for k, v in rows[-1].items():
    print(k, v)

print("total pnl:", sum(r["portfolio_pnl_accounting"] for r in rows))
print("total reward:", sum(r["reward"] for r in rows))
print("roll days:", sum(r["roll_flag"] for r in rows))
print("decision cost:", sum(r["decision_cost"] for r in rows))
print("roll accounting cost:", sum(r["roll_accounting_cost"] for r in rows))