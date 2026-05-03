"""
MAFS5370 Assignment 2 — final submission entry point.

"""
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer


# =============================================================================
# Reproducibility
# =============================================================================
SEED = 59
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
#  Board Definition
# =============================================================================
BLOCKS = [
    (0, 4, 4, 8),    # Level 1
    (4, 8, 2, 6),    # Level 2 left
    (4, 8, 6, 10),   # Level 2 right
    (8, 12, 0, 4),   # Level 3 left
    (8, 12, 4, 8),   # Level 3 middle
    (8, 12, 8, 12),  # Level 3 right
]

ALL_POS: List[Tuple[int, int]] = []
for x1, x2, y1, y2 in BLOCKS:
    for x in range(x1, x2):
        for y in range(y1, y2):
            ALL_POS.append((x, y))

POS_TO_IDX = {p: i for i, p in enumerate(ALL_POS)}
IDX_TO_POS = {i: p for i, p in enumerate(ALL_POS)}
TOTAL_CELLS = len(ALL_POS)  # 96

DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)]


# =============================================================================
# Environment (agent=+1, opponent=-1 random policy)
# =============================================================================
class BoardGameEnv:
    def __init__(self):
        self.board = np.zeros((12, 12), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner: Optional[int] = None

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self._get_state()

    def _get_level(self, x: int) -> int:
        if 0 <= x < 4:
            return 1
        elif 4 <= x < 8:
            return 2
        elif 8 <= x < 12:
            return 3
        return -1

    def _is_valid_pos(self, x: int, y: int) -> bool:
        return (x, y) in POS_TO_IDX and self.board[x, y] == 0

    def valid_actions(self) -> List[int]:
        return [i for i, (x, y) in enumerate(ALL_POS) if self._is_valid_pos(x, y)]

    def _place_piece(self, x: int, y: int) -> bool:
        if self._is_valid_pos(x, y) and not self.game_over:
            self.board[x, y] = self.current_player
            return True
        return False

    def _count_consecutive(self, x: int, y: int, player: int) -> int:
        max_count = 1

        # Horizontal
        count = 1
        cy = y + 1
        while (x, cy) in POS_TO_IDX and self.board[x, cy] == player:
            count += 1
            cy += 1
        cy = y - 1
        while (x, cy) in POS_TO_IDX and self.board[x, cy] == player:
            count += 1
            cy -= 1
        max_count = max(max_count, count)

        # Vertical
        count = 1
        cx = x + 1
        while (cx, y) in POS_TO_IDX and self.board[cx, y] == player:
            count += 1
            cx += 1
        cx = x - 1
        while (cx, y) in POS_TO_IDX and self.board[cx, y] == player:
            count += 1
            cx -= 1
        max_count = max(max_count, count)

        # Diagonal (\)
        count = 1
        cx, cy = x + 1, y + 1
        while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == player:
            count += 1
            cx += 1
            cy += 1
        cx, cy = x - 1, y - 1
        while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == player:
            count += 1
            cx -= 1
            cy -= 1
        max_count = max(max_count, count)

        # Diagonal (/)
        count = 1
        cx, cy = x + 1, y - 1
        while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == player:
            count += 1
            cx += 1
            cy -= 1
        cx, cy = x - 1, y + 1
        while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == player:
            count += 1
            cx -= 1
            cy += 1
        max_count = max(max_count, count)

        return max_count

    def _max_consecutive_on_board(self, player: int) -> int:
        max_count = 0
        for (x, y) in ALL_POS:
            if self.board[x, y] == player:
                max_count = max(max_count, self._count_consecutive(x, y, player))
        return max_count

    @staticmethod
    def _threat_score(max_consecutive: int) -> int:
        if max_consecutive >= 5:
            return 20
        elif max_consecutive == 4:
            return 10
        elif max_consecutive == 3:
            return 3
        return 0

    def _get_graded_reward(self, x: int, y: int) -> float:
        player = self.current_player
        opponent = -player

        # Attack reward
        agent_count = self._count_consecutive(x, y, player)
        attack = 0.0
        if agent_count == 3:
            attack = 5.0
        elif agent_count == 4:
            attack = 15.0
        elif agent_count >= 5:
            attack = 30.0

        # Defense reward (global threat reduction)
        opp_after = self._threat_score(self._max_consecutive_on_board(opponent))

        self.board[x, y] = 0
        opp_before = self._threat_score(self._max_consecutive_on_board(opponent))
        self.board[x, y] = player

        defense = float(opp_before - opp_after)
        defense = max(-5.0, min(10.0, defense))

        return attack + defense

    def _check_win(self) -> bool:
        p = self.current_player
        for (x, y) in ALL_POS:
            if self.board[x, y] != p:
                continue

            # Horizontal 4
            count = 1
            cy = y + 1
            while (x, cy) in POS_TO_IDX and self.board[x, cy] == p:
                count += 1
                if count >= 4:
                    return True
                cy += 1

            # Vertical 4 + cross levels
            count = 1
            levels = {self._get_level(x)}
            cx = x + 1
            while (cx, y) in POS_TO_IDX and self.board[cx, y] == p:
                count += 1
                levels.add(self._get_level(cx))
                if count >= 4 and len(levels) > 1:
                    return True
                cx += 1

            # Diagonal (\) 5
            count = 1
            cx, cy = x + 1, y + 1
            while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == p:
                count += 1
                if count >= 5:
                    return True
                cx += 1
                cy += 1

            # Diagonal (/) 5
            count = 1
            cx, cy = x + 1, y - 1
            while (cx, cy) in POS_TO_IDX and self.board[cx, cy] == p:
                count += 1
                if count >= 5:
                    return True
                cx += 1
                cy -= 1

        return False

    def _check_draw(self) -> bool:
        return all(not self._is_valid_pos(x, y) for (x, y) in ALL_POS)

    def _stochastic_place(self, action: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
        x0, y0 = IDX_TO_POS[action]
        if not self._is_valid_pos(x0, y0):
            return False, None

        if random.random() < 0.5:
            ok = self._place_piece(x0, y0)
            return ok, (x0, y0) if ok else None

        dx, dy = random.choice(DIRS)
        nx, ny = x0 + dx, y0 + dy
        if self._place_piece(nx, ny):
            return True, (nx, ny)
        return False, None

    def _get_state(self) -> np.ndarray:
        return np.array([self.board[p] for p in ALL_POS], dtype=np.float32)

    def step_agent(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        One full environment step from agent perspective:
        1) agent (+1) acts
        2) if game not done, opponent (-1) random acts
        """
        if self.game_over:
            return self._get_state(), 0.0, True, {"info": "Game over"}

        reward = 0.0

        # ---------------- Agent turn ----------------
        self.current_player = 1
        x0, y0 = IDX_TO_POS[action]
        if not self._is_valid_pos(x0, y0):
            return self._get_state(), -2.0, False, {"error": "Invalid action"}

        placed, final_pos = self._stochastic_place(action)
        if not placed:
            reward += -1.0  # move forfeited
        else:
            fx, fy = final_pos
            reward += self._get_graded_reward(fx, fy)

            if self._check_win():
                self.game_over = True
                self.winner = 1
                reward += 100.0
                return self._get_state(), reward, True, {"winner": 1}

            if self._check_draw():
                self.game_over = True
                self.winner = 0
                return self._get_state(), reward, True, {"winner": 0}

        # ---------------- Opponent turn (random) ----------------
        self.current_player = -1
        opp_valid = self.valid_actions()
        if not opp_valid:
            self.game_over = True
            self.winner = 0
            return self._get_state(), reward, True, {"winner": 0}

        opp_action = random.choice(opp_valid)
        opp_placed, opp_pos = self._stochastic_place(opp_action)
        if opp_placed:
            ox, oy = opp_pos
            opp_prog = self._get_graded_reward(ox, oy)
            reward -= 0.25 * max(0.0, opp_prog)

            if self._check_win():
                self.game_over = True
                self.winner = -1
                reward += -100.0
                return self._get_state(), reward, True, {"winner": -1}

            if self._check_draw():
                self.game_over = True
                self.winner = 0
                return self._get_state(), reward, True, {"winner": 0}

        self.current_player = 1
        return self._get_state(), reward, False, {}


# =============================================================================
# Torch DQN Network
# =============================================================================
class QNet(nn.Module):
    def __init__(self, state_dim=96, action_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainConfig:
    episodes: int = 50000
    batch_size: int = 64
    gamma: float = 0.95
    lr: float = 5e-5
    lr_late: float = 1e-5
    lr_decay_episode: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    buffer_size: int = 100000
    tau: float = 0.001
    log_interval: int = 100
    eval_interval: int = 1000
    eval_episodes: int = 200
    target_sync_interval: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def masked_argmax(q_values: torch.Tensor, valid_actions: List[int]) -> int:
    mask = torch.full_like(q_values, -torch.inf)
    mask[valid_actions] = q_values[valid_actions]
    return int(torch.argmax(mask).item())


def soft_update(target: nn.Module, online: nn.Module, tau: float):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


@torch.no_grad()
def evaluate_stats(policy_net: nn.Module, episodes: int = 200, device: Optional[str] = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = BoardGameEnv()
    policy_net = policy_net.to(device)
    policy_net.eval()

    wins = losses = draws = 0
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        info = {}

        while not done:
            valid_actions = env.valid_actions()
            if not valid_actions:
                break
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = policy_net(s).squeeze(0)
            action = masked_argmax(q, valid_actions)
            state, reward, done, info = env.step_agent(action)
            ep_r += reward

        total_rewards.append(ep_r)
        w = info.get("winner", 0)
        if w == 1:
            wins += 1
        elif w == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "draw_rate": draws / episodes,
        "avg_reward": float(np.mean(total_rewards)),
    }


def train_torchrl(cfg: TrainConfig):
    device = torch.device(cfg.device)

    env = BoardGameEnv()
    q_net = QNet().to(device)
    target_net = QNet().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr, weight_decay=1e-6)
    loss_fn = nn.SmoothL1Loss()

    storage = LazyTensorStorage(cfg.buffer_size, device=torch.device("cpu"))
    replay_buffer = TensorDictReplayBuffer(storage=storage, batch_size=cfg.batch_size)

    epsilon = cfg.epsilon_start
    best_episode_reward = -float("inf")
    train_steps = 0
    start_time = time.time()

    reward_window = deque(maxlen=500)
    win_window = deque(maxlen=500)
    loss_window = deque(maxlen=5000)

    print(f"Training on device: {device}")
    print("Start TorchRL-DoubleDQN training...")

    for ep in range(1, cfg.episodes + 1):
        if ep == cfg.lr_decay_episode:
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr_late
            print(f"[LR] Decayed learning rate to {cfg.lr_late}")

        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_updates = 0

        while not done:
            valid_actions = env.valid_actions()
            if not valid_actions:
                break

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q = q_net(s).squeeze(0)
                    action = masked_argmax(q, valid_actions)

            next_state, reward, done, _ = env.step_agent(action)
            ep_reward += reward

            next_valid_actions = env.valid_actions() if not done else []
            next_valid_mask = torch.zeros(TOTAL_CELLS, dtype=torch.bool)
            if next_valid_actions:
                next_valid_mask[next_valid_actions] = True

            td = TensorDict(
                {
                    "state": torch.tensor(state, dtype=torch.float32),
                    "action": torch.tensor(action, dtype=torch.long),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_state": torch.tensor(next_state, dtype=torch.float32),
                    "done": torch.tensor(float(done), dtype=torch.float32),
                    "next_valid_mask": next_valid_mask,
                },
                batch_size=[],
            )
            replay_buffer.add(td)
            state = next_state

            if len(replay_buffer) >= cfg.batch_size:
                batch = replay_buffer.sample()
                states = batch["state"].to(device)
                actions = batch["action"].to(device)
                rewards = batch["reward"].to(device)
                next_states = batch["next_state"].to(device)
                dones = batch["done"].to(device)
                next_valid_mask = batch["next_valid_mask"].to(device)

                q_pred = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                q_pred = torch.clamp(q_pred, -150.0, 150.0)

                # Double DQN target with valid-action mask
                with torch.no_grad():
                    next_q_online = q_net(next_states)       # selection
                    next_q_target = target_net(next_states)  # evaluation

                    neg_inf = torch.finfo(next_q_online.dtype).min
                    masked_online = next_q_online.masked_fill(~next_valid_mask, neg_inf)
                    next_actions = masked_online.argmax(dim=1, keepdim=True)

                    max_next_q = next_q_target.gather(1, next_actions).squeeze(1)

                    has_valid = next_valid_mask.any(dim=1)
                    max_next_q = torch.where(has_valid, max_next_q, torch.zeros_like(max_next_q))

                    target = rewards + cfg.gamma * max_next_q * (1.0 - dones)
                    target = torch.clamp(target, -150.0, 150.0)

                loss = loss_fn(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss_sum += float(loss.item())
                ep_updates += 1
                train_steps += 1
                loss_window.append(float(loss.item()))

                if train_steps % cfg.target_sync_interval == 0:
                    soft_update(target_net, q_net, cfg.tau)

        reward_window.append(ep_reward)
        if env.winner == 1:
            win_window.append(1)
        elif env.winner == -1:
            win_window.append(-1)
        else:
            win_window.append(0)

        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward

        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

        if ep % cfg.log_interval == 0:
            avg_loss = (ep_loss_sum / ep_updates) if ep_updates > 0 else 0.0
            elapsed = time.time() - start_time

            avg_r_500 = float(np.mean(reward_window)) if reward_window else 0.0
            p25_r = float(np.percentile(reward_window, 25)) if reward_window else 0.0
            p50_r = float(np.percentile(reward_window, 50)) if reward_window else 0.0
            p75_r = float(np.percentile(reward_window, 75)) if reward_window else 0.0

            if win_window:
                win_rate = sum(1 for x in win_window if x == 1) / len(win_window)
                loss_rate = sum(1 for x in win_window if x == -1) / len(win_window)
                draw_rate = sum(1 for x in win_window if x == 0) / len(win_window)
            else:
                win_rate = loss_rate = draw_rate = 0.0

            loss_ma = float(np.mean(loss_window)) if loss_window else 0.0
            loss_std = float(np.std(loss_window)) if len(loss_window) > 1 else 0.0

            print(
                f"Ep:{ep:6d} | R:{ep_reward:8.2f} | R500:{avg_r_500:7.2f} "
                f"| R[p25/p50/p75]:{p25_r:6.1f}/{p50_r:6.1f}/{p75_r:6.1f} "
                f"| W/L/D:{win_rate:5.1%}/{loss_rate:5.1%}/{draw_rate:5.1%} "
                f"| Loss(ep):{avg_loss:.5f} | Loss(ma):{loss_ma:.5f}±{loss_std:.5f} "
                f"| Eps:{epsilon:.3f} | LR:{optimizer.param_groups[0]['lr']:.1e} "
                f"| Buf:{len(replay_buffer):6d} | T:{elapsed:.1f}s"
            )

        if cfg.eval_interval > 0 and ep % cfg.eval_interval == 0:
            stats = evaluate_stats(q_net, episodes=cfg.eval_episodes, device=cfg.device)
            print(
                f"[Eval@{ep}] "
                f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
                f"| WR/LR/DR:{stats['win_rate']:.1%}/{stats['loss_rate']:.1%}/{stats['draw_rate']:.1%} "
                f"| AvgR:{stats['avg_reward']:.2f}"
            )

    print("\nTraining completed.")
    print(f"Best episode reward: {best_episode_reward:.2f}")

    return q_net, target_net


@torch.no_grad()
def evaluate(policy_net: nn.Module, episodes: int = 50, device: Optional[str] = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = BoardGameEnv()
    policy_net = policy_net.to(device)
    policy_net.eval()

    wins = losses = draws = 0
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        info = {}

        while not done:
            valid_actions = env.valid_actions()
            if not valid_actions:
                break
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = policy_net(s).squeeze(0)
            action = masked_argmax(q, valid_actions)
            state, reward, done, info = env.step_agent(action)
            ep_r += reward

        total_rewards.append(ep_r)
        w = info.get("winner", 0)
        if w == 1:
            wins += 1
        elif w == -1:
            losses += 1
        else:
            draws += 1

    print("\nEvaluation:")
    print(f"Wins={wins}, Losses={losses}, Draws={draws}")
    print(f"AvgReward={np.mean(total_rewards):.3f}")


if __name__ == "__main__":
    config = TrainConfig(
        episodes=20000,
        batch_size=64,
        gamma=0.95,
        lr=5e-5,
        lr_late=1e-5,
        lr_decay_episode=10000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        buffer_size=100000,
        tau=0.001,
        log_interval=100,
        eval_interval=1000,
        eval_episodes=200,
        target_sync_interval=4,
    )

    q_net, target_net = train_torchrl(config)
    evaluate(q_net, episodes=200, device=config.device)

    torch.save(q_net.state_dict(), "dqn_torchrl.pt")
    print("Saved model to dqn_torchrl.pt")
