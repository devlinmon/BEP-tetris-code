import csv
import random
from statistics import mean

try:
    import torch
except ImportError:
    torch = None

from board import Board
from game import game
from rl_agent import RLAgent


def game_score(lines_cleared: int) -> int:
    return 100 * (lines_cleared ** 2)


def compute_reward(agent, new_board, lines_cleared, cleared_cells):
    try:
        return agent.reward_function(new_board, lines_cleared, cleared_cells)
    except TypeError:
        return agent.reward_function(new_board, lines_cleared)


def train_agent(
    episodes: int = 1000,
    max_moves: int = 1000,
    save_path: str = "tetris_rl_agent.pt",
    log_path: str = "training_log.csv",
    seed: int = 42,
    use_bomb: bool = False,
    bomb_probability: float = 0.03,
):
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    sim = game(use_bomb=use_bomb, bomb_probability=bomb_probability)
    agent = RLAgent()

    episode_rewards = []      
    episode_game_scores = []  
    episode_lines = []
    episode_moves = []
    losses = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "reward",
                "game_score",
                "lines",
                "moves",
                "epsilon",
                "avg_reward_25",
                "avg_game_score_25",
                "avg_lines_25",
                "avg_moves_25",
            ],
        )
        writer.writeheader()

        for episode in range(1, episodes + 1):
            board = Board()
            total_reward = 0.0
            total_game_score = 0
            total_lines = 0
            moves_played = 0
            done = False

            while not done and moves_played < max_moves:
                current_block = sim.get_random_block()
                block_class = current_block.__class__

                candidates = agent.get_candidate_moves(board, block_class)
                if not candidates:
                    done = True
                    break

                chosen = agent.select_action(board, block_class, training=True)
                if chosen is None:
                    done = True
                    break

        
                rotation, col = chosen[0]
                new_board = chosen[1]
                lines_cleared = chosen[2]
                state_features = chosen[3]
                cleared_cells = chosen[4] if len(chosen) > 4 else 0

                reward = compute_reward(agent, new_board, lines_cleared, cleared_cells)

                
                board.grid = [row[:] for row in new_board.grid]

                total_reward += reward
                total_game_score += game_score(lines_cleared)
                total_lines += lines_cleared
                moves_played += 1

                next_block = sim.get_random_block()
                next_block_class = next_block.__class__
                next_candidates = agent.get_candidate_moves(board, next_block_class)

                if not next_candidates:
                    next_state = None
                    done = True
                else:
                    with_best = agent.select_action(board, next_block_class, training=False)
                    next_state = with_best[3] if with_best is not None else None

                agent.store_transition(
                    state=state_features,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            agent.decay_epsilon()

            episode_rewards.append(total_reward)
            episode_game_scores.append(total_game_score)
            episode_lines.append(total_lines)
            episode_moves.append(moves_played)

            avg_reward_25 = mean(episode_rewards[-25:])
            avg_game_score_25 = mean(episode_game_scores[-25:])
            avg_lines_25 = mean(episode_lines[-25:])
            avg_moves_25 = mean(episode_moves[-25:])

            writer.writerow(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "game_score": total_game_score,
                    "lines": total_lines,
                    "moves": moves_played,
                    "epsilon": agent.epsilon,
                    "avg_reward_25": avg_reward_25,
                    "avg_game_score_25": avg_game_score_25,
                    "avg_lines_25": avg_lines_25,
                    "avg_moves_25": avg_moves_25,
                }
            )

            if episode % 25 == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"avg_reward={avg_reward_25:8.2f} | "
                    f"avg_game_score={avg_game_score_25:8.2f} | "
                    f"avg_lines={avg_lines_25:6.2f} | "
                    f"avg_moves={avg_moves_25:6.2f} | "
                    f"epsilon={agent.epsilon:.3f}"
                )

    agent.save(save_path)
    print(f"\nSaved model to {save_path}")
    print(f"Saved training log to {log_path}")

    if losses:
        print(f"Final average training loss: {mean(losses[-100:]):.4f}")

    return agent


if __name__ == "__main__":
    train_agent(
        episodes=1500,
        max_moves=1000,
        save_path="final_tetris_rl_agent.pt",
        log_path="final_training_log.csv",
        seed=42,
        use_bomb=False,
    )
