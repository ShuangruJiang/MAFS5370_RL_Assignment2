"""
Monte Carlo comparison of stochastic placement on an empty board.

- OLD: neighbor branch shuffles 8 offsets and tries until one succeeds
  (legacy Assign2.5-style implementation, kept here for ablation only).
- NEW: neighbor branch samples one offset uniformly — matches `Assign2.py`
  (`BoardGameEnv._stochastic_place`).

Loads symbols from the submission module `Assign2.py` (no dependency on
Assign2.5_stochastic_fix.py).
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ASSIGN2 = ROOT / "Assign2.py"
_spec = importlib.util.spec_from_file_location("assign2_mod", _ASSIGN2)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

ALL_POS = _mod.ALL_POS
BoardGameEnv = _mod.BoardGameEnv
DIRS = _mod.DIRS
IDX_TO_POS = _mod.IDX_TO_POS


def neighbor_branch_old(env: BoardGameEnv, x0: int, y0: int) -> bool:
    neighbors = [(x0 + dx, y0 + dy) for dx, dy in DIRS]
    random.shuffle(neighbors)
    for nx, ny in neighbors:
        if env._place_piece(nx, ny):
            return True
    return False


def neighbor_branch_new(env: BoardGameEnv, x0: int, y0: int) -> bool:
    dx, dy = random.choice(DIRS)
    nx, ny = x0 + dx, y0 + dy
    return bool(env._place_piece(nx, ny))


def estimate_conditional_success(
    neighbor_fn, trials: int, seed: int
) -> tuple[int, int, float]:
    """Returns (successes, total_valid_starts, rate)."""
    random.seed(seed)
    ok = total = 0
    for _ in range(trials):
        env = BoardGameEnv()
        env.reset()
        env.current_player = 1
        action = random.randrange(len(ALL_POS))
        x0, y0 = IDX_TO_POS[action]
        if not env._is_valid_pos(x0, y0):
            continue
        total += 1
        if neighbor_fn(env, x0, y0):
            ok += 1
    rate = ok / total if total else 0.0
    return ok, total, rate


def full_move_forfeit_rate(
    use_old: bool, trials: int, seed: int
) -> tuple[int, int, float]:
    """Simulate full _stochastic_place logic including 50% center branch."""
    random.seed(seed)
    forfeits = total = 0
    for _ in range(trials):
        env = BoardGameEnv()
        env.reset()
        env.current_player = 1
        action = random.randrange(len(ALL_POS))
        x0, y0 = IDX_TO_POS[action]
        if not env._is_valid_pos(x0, y0):
            continue
        total += 1
        placed = False
        if random.random() < 0.5:
            placed = env._place_piece(x0, y0)
        else:
            if use_old:
                placed = neighbor_branch_old(env, x0, y0)
            else:
                placed = neighbor_branch_new(env, x0, y0)
        if not placed:
            forfeits += 1
    rate = forfeits / total if total else 0.0
    return forfeits, total, rate


def main():
    n = 80_000
    seed = 2026
    o_ok, o_tot, o_rate = estimate_conditional_success(neighbor_branch_old, n, seed)
    n_ok, n_tot, n_rate = estimate_conditional_success(neighbor_branch_new, n, seed + 1)
    f_old, t_old, r_old = full_move_forfeit_rate(True, n, seed + 2)
    f_new, t_new, r_new = full_move_forfeit_rate(False, n, seed + 3)

    print("=== Neighbor branch only P(success | empty cell chosen) ===")
    print(f"OLD: {o_ok}/{o_tot} = {o_rate:.4f}")
    print(f"NEW (Assign2.py rule): {n_ok}/{n_tot} = {n_rate:.4f}")
    print("=== Full trial (50% center + 50% neighbor), P(forfeit) ===")
    print(f"OLD: {f_old}/{t_old} = {r_old:.4f}")
    print(f"NEW (Assign2.py rule): {f_new}/{t_new} = {r_new:.4f}")


if __name__ == "__main__":
    main()
