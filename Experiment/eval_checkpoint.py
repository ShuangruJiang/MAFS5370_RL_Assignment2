"""Evaluate a saved state_dict using `Assign2.py` (BoardGameEnv + QNet)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_ASSIGN2 = ROOT / "Assign2.py"
_spec = importlib.util.spec_from_file_location("assign2_mod", _ASSIGN2)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

QNet = _mod.QNet
evaluate_stats = _mod.evaluate_stats


def main():
    for name in ("pyramid_dqn_torchrl.pt", "dqn_torchrl_stochastic_fix.pt"):
        ckpt = ROOT / name
        if not ckpt.is_file():
            continue
        net = QNet()
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        stats = evaluate_stats(net, episodes=200, device="cpu")
        print("Checkpoint:", ckpt.name)
        print(stats)
        return
    print(f"No checkpoint found in {ROOT} (tried pyramid_dqn_torchrl.pt, dqn_torchrl_stochastic_fix.pt)")


if __name__ == "__main__":
    main()
