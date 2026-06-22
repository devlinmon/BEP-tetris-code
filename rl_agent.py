import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from board import Board
import board
from game import game


@dataclass
class Transition:
    state: torch.Tensor
    reward: float
    next_state: Optional[torch.Tensor]
    done: bool


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class RLAgent:
    def __init__(
        self,
        input_dim: int = 8,
        gamma: float = 0.99,
        lr: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        target_update_freq: int = 250,
        batch_size: int = 64,
        replay_capacity: int = 50000,
        device: Optional[str] = None,
        reward_mode: str = "tetris_survival",
        feature_mode: str = "high",
    ):
        self.input_dim = input_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.train_steps = 0

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = QNetwork(input_dim=input_dim).to(self.device)
        self.target_net = QNetwork(input_dim=input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.reward_mode = reward_mode
        self.feature_mode = feature_mode
        self.sim = game()
    def count_row_transitions(self, board: Board) -> int:
        transitions = 0

        for row in board.grid:
            previous_filled = True 

            for cell in row:
                current_filled = cell != 0
                if current_filled != previous_filled:
                    transitions += 1
                previous_filled = current_filled

            if previous_filled is False:  
                transitions += 1

        return transitions


    def count_column_transitions(self, board: Board) -> int:
        transitions = 0

        for col in range(board.nr_cols):
            previous_filled = True  

            for row in reversed(range(board.nr_rows)):
                current_filled = board.grid[row][col] != 0
                if current_filled != previous_filled:
                    transitions += 1
                previous_filled = current_filled

            if previous_filled is False:  
                transitions += 1

        return transitions


    def get_deepest_well(self, heights: list[int]) -> int:
        deepest = 0

        for i in range(len(heights)):
            left = heights[i - 1] if i > 0 else 20
            right = heights[i + 1] if i < len(heights) - 1 else 20

            well_depth = min(left, right) - heights[i]

            if well_depth > deepest:
                deepest = well_depth

        return deepest

    def board_to_features(self, board: Board, lines_cleared: int = 0) -> torch.Tensor:
        heights = self.sim.get_column_heights(board)
        aggregate_height = sum(heights)
        holes = self.sim.count_holes(board)
        bumpiness = self.sim.get_bumpiness(heights)
        max_height = max(heights) if heights else 0
        completed_lines = lines_cleared
        row_transitions = self.count_row_transitions(board)
        column_transitions = self.count_column_transitions(board)
        deepest_well = self.get_deepest_well(heights)

        if self.feature_mode == "low":
            features = [
                aggregate_height / 200.0,
                holes / 200.0,
                bumpiness / 200.0,
            ]

        elif self.feature_mode == "medium":
            features = [
                aggregate_height / 200.0,
                holes / 200.0,
                bumpiness / 200.0,
                max_height / 20.0,
                completed_lines / 4.0,
            ]

        elif self.feature_mode == "high":
            features = [
                aggregate_height / 200.0,
                holes / 200.0,
                bumpiness / 200.0,
                max_height / 20.0,
                completed_lines / 4.0,
                row_transitions / 200.0,
                column_transitions / 400.0,
                deepest_well / 20.0,
        ]
        elif self.feature_mode == "Dellacherie":
            features = [
            row_transitions / 200.0,
            column_transitions / 400.0,
            holes / 200.0,
            deepest_well / 20.0,
            completed_lines / 4.0,
            max_height / 20.0,
        ]

        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def reward_function(self, board: Board, lines_cleared: int, cleared_cells: int = 0) -> float:
        heights = self.sim.get_column_heights(board)
        aggregate_height = sum(heights)
        holes = self.sim.count_holes(board)
        bumpiness = self.sim.get_bumpiness(heights)

        if self.reward_mode == "baseline":
            reward = (
                1.0
                + 20.0 * lines_cleared
                - 0.1 * aggregate_height
                - 0.3 * holes
                - 0.1 * bumpiness
            )

        elif self.reward_mode == "strong_holes":
            reward = (
                1.0
                + 20.0 * lines_cleared
                - 0.1 * aggregate_height
                - 0.6 * holes
                - 0.1 * bumpiness
            )

        elif self.reward_mode == "tetris_bonus":
            line_bonus = [0, 40, 100, 300, 1200][lines_cleared]

            reward = (
                1.0
                + line_bonus
                - 0.1 * aggregate_height
                - 0.3 * holes
                - 0.1 * bumpiness
            )
        elif self.reward_mode == "survival_bonus":
            reward = (
                5.0                      
                + 20.0 * lines_cleared
                - 0.1 * aggregate_height
                - 0.3 * holes
                - 0.1 * bumpiness
            )
        elif self.reward_mode == "tetris_survival":
            line_bonus = [0, 40, 100, 300, 1200][lines_cleared]

            reward = (
                20.0
                + line_bonus
                - 0.1 * aggregate_height
                - 0.3 * holes
                - 0.1 * bumpiness
            )
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return reward
    def get_valid_columns_for_rotation(self, board: Board, block_class, rotation):
        block = block_class()
        cells = block.cells[rotation]

        min_x = min(cell.x for cell in cells)
        max_x = max(cell.x for cell in cells)

        start_col = -min_x
        end_col = board.nr_cols - max_x

        return range(start_col, end_col)

    def get_candidate_moves(
        self,
        board: Board,
        block_class,
    ) -> List[Tuple[Tuple[int, int], Board, int, torch.Tensor]]:
        candidates = []

        for rotation in block_class().cells.keys():
            for col in self.get_valid_columns_for_rotation(board, block_class, rotation):
                result = self.sim.simulate_move(board, block_class, rotation, col)
                if result is None:
                    continue

                new_board, _, cleared_lines, cleared_cells = result
                state_features = self.board_to_features(new_board, cleared_lines)
                candidates.append(((rotation, col), new_board, cleared_lines, state_features, cleared_cells))

        return candidates

    def select_action(
        self,
        board: Board,
        block_class,
        training: bool = True,
    ):
        candidates = self.get_candidate_moves(board, block_class)
        if not candidates:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(candidates)

        with torch.no_grad():
            state_batch = torch.stack([c[3] for c in candidates]) 
            q_values = self.policy_net(state_batch).squeeze(-1)
            best_idx = torch.argmax(q_values).item()

        return candidates[best_idx]

    def store_transition(
        self,
        state: torch.Tensor,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
    ) -> None:
        self.replay_buffer.push(
            Transition(
                state=state.detach().clone(),
                reward=reward,
                next_state=None if next_state is None else next_state.detach().clone(),
                done=done,
            )
        )

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.stack([t.state for t in batch]).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        next_states = []
        non_terminal_mask = []
        for t in batch:
            if t.next_state is None:
                non_terminal_mask.append(False)
            else:
                non_terminal_mask.append(True)
                next_states.append(t.next_state)

        current_q = self.policy_net(states).squeeze(-1)

        next_q = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        if next_states:
            next_states_tensor = torch.stack(next_states).to(self.device)
            target_vals = self.target_net(next_states_tensor).squeeze(-1)

            idx = 0
            for i, is_non_terminal in enumerate(non_terminal_mask):
                if is_non_terminal:
                    next_q[i] = target_vals[idx]
                    idx += 1

        target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)