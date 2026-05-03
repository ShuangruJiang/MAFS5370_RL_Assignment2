"""Short training run for reporting — imports `Assign2.py` (final submission)."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ASSIGN2 = ROOT / "Assign2.py"
_spec = importlib.util.spec_from_file_location("assign2_mod", _ASSIGN2)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

TrainConfig = _mod.TrainConfig
train_torchrl = _mod.train_torchrl
evaluate_stats = _mod.evaluate_stats


def main():
    cfg = TrainConfig(
        episodes=400,
        batch_size=64,
        gamma=0.95,
        lr=5e-5,
        lr_late=1e-5,
        lr_decay_episode=10_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        buffer_size=50_000,
        tau=0.001,
        log_interval=100,
        eval_interval=0,
        eval_episodes=120,
        target_sync_interval=4,
        device="cpu",
    )
    t0 = time.time()
    q_net, _ = train_torchrl(cfg)
    dt = time.time() - t0
    stats = evaluate_stats(q_net, episodes=120, device="cpu")
    print(f"=== wall_clock_s={dt:.1f} ===")
    print(stats)


if __name__ == "__main__":
    main()
