import random
import csv
from statistics import mean

import torch

from board import Board
from game import game
from rl_agent import RLAgent


def game_score(lines_cleared: int) -> int:
    return 100 * (lines_cleared ** 2)


def train_one_reward(
    reward_mode: str,
    episodes: int = 500,
    max_moves: int = 1000,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    sim = game(use_bomb=False)
    agent = RLAgent(reward_mode=reward_mode)

    episode_scores = []
    episode_lines = []
    episode_moves = []

    for episode in range(1, episodes + 1):
        board = Board()
        total_reward = 0.0
        total_lines = 0
        moves_played = 0
        done = False

        while not done and moves_played < max_moves:
            current_block = sim.get_random_block()
            block_class = current_block.__class__

            chosen = agent.select_action(board, block_class, training=True)

            if chosen is None:
                break

            rotation, col = chosen[0]

            result = sim.simulate_move(board, block_class, rotation, col)

            if result is None:
                break

            new_board, _, lines_cleared, _ = result

            reward = agent.reward_function(new_board, lines_cleared)

            board.grid = [row[:] for row in new_board.grid]

            next_block = sim.get_random_block()
            next_block_class = next_block.__class__
            next_choice = agent.select_action(board, next_block_class, training=False)

            if next_choice is None:
                next_state = None
                done = True
            else:
                next_state = next_choice[3]

            agent.store_transition(
                state=chosen[3],
                reward=reward,
                next_state=next_state,
                done=done,
            )

            agent.train_step()

            total_reward += reward
            total_lines += lines_cleared
            moves_played += 1

        agent.decay_epsilon()

        episode_scores.append(total_reward)
        episode_lines.append(total_lines)
        episode_moves.append(moves_played)

        if episode % 25 == 0:
            print(
                f"{reward_mode} | Episode {episode:4d} | "
                f"avg_reward={mean(episode_scores[-25:]):8.2f} | "
                f"avg_lines={mean(episode_lines[-25:]):6.2f} | "
                f"avg_moves={mean(episode_moves[-25:]):6.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    model_path = f"tetris_rl_{reward_mode}.pt"
    agent.save(model_path)

    return model_path


def evaluate_model(
    model_path: str,
    reward_mode: str,
    num_games: int = 100,
    max_moves: int = 1000,
):
    sim = game(use_bomb=False)
    agent = RLAgent(reward_mode=reward_mode)
    agent.load(model_path)
    agent.epsilon = 0.0

    scores = []
    lines_list = []
    moves_list = []

    for _ in range(num_games):
        board = Board()
        total_score = 0
        total_lines = 0
        moves_played = 0

        while moves_played < max_moves:
            current_block = sim.get_random_block()
            block_class = current_block.__class__

            chosen = agent.select_action(board, block_class, training=False)

            if chosen is None:
                break

            rotation, col = chosen[0]

            result = sim.simulate_move(board, block_class, rotation, col)

            if result is None:
                break

            new_board, _, lines_cleared, _ = result

            board.grid = [row[:] for row in new_board.grid]

            total_score += game_score(lines_cleared)
            total_lines += lines_cleared
            moves_played += 1

        scores.append(total_score)
        lines_list.append(total_lines)
        moves_list.append(moves_played)

    return {
        "reward_mode": reward_mode,
        "avg_score": mean(scores),
        "avg_lines": mean(lines_list),
        "avg_moves": mean(moves_list),
    }


def run_rq2_experiment():
    reward_modes = [
        "baseline",
        "strong_holes",
        "tetris_bonus",
    ]

    results = []

    for reward_mode in reward_modes:
        print(f"\nTraining reward mode: {reward_mode}")
        model_path = train_one_reward(reward_mode)

        print(f"Evaluating reward mode: {reward_mode}")
        result = evaluate_model(model_path, reward_mode)
        results.append(result)

    with open("rq2_reward_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["reward_mode", "avg_score", "avg_lines", "avg_moves"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nRQ2 reward design results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    run_rq2_experiment()