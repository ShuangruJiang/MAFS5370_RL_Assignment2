# MAFS5370 Assignment 2 — Experiments and Results

**Project:** Pyramid “super tic-tac-toe” board, Double DQN with TorchRL replay, agent (+1) versus uniform-random opponent (−1).  
**Final code / submission entry:** `Assign2.py` — includes assignment-compliant stochastic placement (50% on chosen empty cell; otherwise **one** uniformly random 8-neighbor trial; forfeit if illegal), `BoardGameEnv`, `QNet`, `train_torchrl`, and evaluation helpers.  
**Report date:** 2026-05-03  

---

## 1. Objective

Document empirical behavior of the **stochastic move rule** (legacy sequential-neighbor retry **vs.** single-neighbor trial as implemented in `Assign2.py`) and report **training plus evaluation** metrics for the TorchRL Double DQN pipeline under the **final** environment dynamics.

---

## 2. Experimental setup

| Item | Setting |
|------|---------|
| State | 96-D float vector of valid pyramid cells |
| Actions | 96 discrete indices; invalid moves masked in policy and bootstrap |
| Opponent | Uniform random among legal actions each opponent ply |
| Algorithm | Double DQN, Huber (`SmoothL1`) loss, Q/target clamp \([-150,150]\), grad clip 1.0, Polyak \(\tau=0.001\) toward target every 4 learner steps |
| Replay | `TensorDictReplayBuffer` + `LazyTensorStorage` on CPU (TorchRL) |
| Exploration | \(\varepsilon\)-greedy; short run used \(\varepsilon \leftarrow \max(0.05,\,0.997\,\varepsilon)\) per episode |
| Seeds | `SEED=59` in `Assign2.py`; placement benchmark uses its own RNG seeds |
| Hardware (this run) | Apple host, **CPU** training for the short experiment (`device="cpu"`) |
| Software | Python 3.13, PyTorch + TorchRL from project `.venv` |


| Script | Purpose |
|--------|---------|
| `benchmark_placement.py` | Monte Carlo: OLD neighbor-retry vs NEW (same rule as `Assign2.py`) |
| `short_train_assign2.py` | Abbreviated training + eval on `Assign2` pipeline |
| `eval_checkpoint.py` | Load `pyramid_dqn_torchrl.pt` (or legacy `dqn_torchrl_stochastic_fix.pt`) and `evaluate_stats` |

**Full training / default hyperparameters:** run `python Assign2.py (saves `dqn_torchrl.pt`).

---

## 3. Experiment A — Stochastic placement mechanics

**Goal:** Quantify how a **legacy** neighbor branch (shuffle 8 offsets, try until one succeeds) differs from the **submission** rule in `Assign2.py` (single random neighbor; forfeit if not legal).

**Procedure:** On an **empty** board, sample a uniform random **legal** cell as the chosen action, then execute **only** the “not placed on chosen cell” branch (neighbor branch). A second block estimates **overall** forfeit probability when the 50% / 50% coin is included.

**Trials:** 80,000 valid random starts per metric (Monte Carlo).

### Results

| Metric | OLD (sequential neighbors) | NEW (`Assign2.py`) |
|--------|---------------------------|----------------------|
| \(P(\text{place} \mid \text{neighbor branch})\) | **1.0000** (80,000 / 80,000) | **0.8291** (66,329 / 80,000) |
| \(P(\text{forfeit} \mid \text{full random trial})\)* | **0.0000** (0 / 80,000) | **0.0868** (6,944 / 80,000) |

\*Full trial: with probability \(1/2\) place on chosen cell; with probability \(1/2\) enter neighbor branch as above.

### Interpretation

- Under the **legacy** retry rule on an empty board, the neighbor branch **always** finds a legal neighbor in this geometry, hence **100%** placement and **0%** forfeit in the Monte Carlo.
- Under **`Assign2.py`**, **~17%** of neighbor-branch attempts fail (off-pyramid or occupied), and **~8.7%** of all stochastic trials forfeit—consistent with the assignment’s stricter “single draw” behavior.

---

## 4. Experiment B — Short training run (`Assign2.py`)

**Goal:** Show learning progress and greedy performance versus the random opponent after limited training using the **same** module as submission.

**Procedure:** `short_train_assign2.py` — `TrainConfig` with `episodes=400`, `buffer_size=50_000`, `eval_interval=0`, `epsilon_decay=0.997`, `device="cpu"`. After training, `evaluate_stats` for **120** episodes (greedy masked argmax).

### Training log (selected episodes)

| Episode | Ep return \(R\) | R500 mean | W/L/D (last 500 window)* | \(\varepsilon\) | Buffer size |
|--------:|----------------:|----------:|---------------------------|------------------:|------------:|
| 100 | 148.25 | 32.12 | 55.0% / 45.0% / 0.0% | 0.740 | 2,977 |
| 200 | −103.75 | 59.27 | 66.0% / 34.0% / 0.0% | 0.548 | 5,679 |
| 300 | 141.75 | 73.89 | 72.7% / 27.3% / 0.0% | 0.406 | 7,778 |
| 400 | 143.75 | 83.89 | 76.8% / 23.2% / 0.0% | 0.301 | 9,857 |

\*Window statistics from the trainer: **last up to 500** training episodes’ terminal outcomes, not a separate benchmark opponent.

**Best training episode return:** 219.25  
**Wall clock (train only):** ~29.5 s on CPU for 400 episodes.

### Post-hoc evaluation (120 games, greedy)

| Metric | Value |
|--------|------:|
| Wins | 118 |
| Losses | 2 |
| Draws | 0 |
| Win rate | **98.33%** |
| Loss rate | 1.67% |
| Mean return | **140.47** |

### Interpretation

Short training on the **final** dynamics already yields a **high** win rate vs. the random opponent; episode returns remain noisy over only 400 episodes.

---

## 5. Experiment C — Checkpoint evaluation under `Assign2.py`

**Goal:** Evaluate saved weights with `Assign2.py`’s `QNet` and `BoardGameEnv` (same as submission).

**Procedure:** `eval_checkpoint.py` loads **`dqn_torchrl.pt`** if present (else `dqn_torchrl_stochastic_fix.pt`), **200** episodes, CPU.

### Results (representative run with `dqn_torchrl.pt`)

| Metric | Value |
|--------|------:|
| Wins | 197 |
| Losses | 3 |
| Draws | 0 |
| Win rate | **98.50%** |
| Loss rate | 1.50% |
| Mean return | **144.57** |

### Interpretation

Greedy policy remains strong under the current environment API; for grading, **train and save with `Assign2.py`** so checkpoint provenance matches the submitted source file.

---

