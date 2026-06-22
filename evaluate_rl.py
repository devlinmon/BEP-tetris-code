from statistics import mean
from board import Board
from game import game
from rl_agent import RLAgent
import matplotlib.pyplot as plt

def evaluate_agent(
    model_path: str = "tetris_rl_state_high.pt",
    feature_mode: str = "high",
    num_games: int = 300,
    max_moves: int = 2000,
):
    sim = game(use_bomb=True)
    agent = RLAgent(feature_mode=feature_mode)
    agent.load(model_path)
    agent.epsilon = 0.0

    scores = []
    lines_list = []
    moves_list = []

    bombs_used_list = []
    bomb_cells_list = []

    lines_after_bomb_list = []
    height_reduction_list = []
    hole_reduction_list = []
    tetris_list = []
    holes_created_list = []
    holes_removed_list = []
    total_line_clear_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }
    holes_by_move = [[] for _ in range(max_moves)]

    for _ in range(num_games):

        board = Board()

        total_score = 0.0
        total_lines = 0
        moves_played = 0

        bombs_used = 0
        bomb_cells_cleared = 0
        tetrises = 0
        holes_created = 0
        holes_removed = 0
        tetrises = 0
        holes_created = 0
        holes_removed = 0

        line_clear_counts = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }

        bomb_lines_cleared = 0

        total_height_reduction = 0
        total_hole_reduction = 0

        while moves_played < max_moves:

            current_block = sim.get_random_block()
            block_class = current_block.__class__

            chosen = agent.select_action(
                board,
                block_class,
                training=False
            )

            if chosen is None:
                break

            (rotation, col) = chosen[0]

            old_heights = sim.get_column_heights(board)
            old_height = sum(old_heights)

            old_holes = sim.count_holes(board)

            result = sim.simulate_move(
                board,
                block_class,
                rotation,
                col
            )

            if result is None:
                break

            new_board, placed_block, lines_cleared, cleared_cells = result

            new_heights = sim.get_column_heights(new_board)
            new_height = sum(new_heights)

            new_holes = sim.count_holes(new_board)
            if lines_cleared == 4:
                tetrises += 1
            hole_difference = new_holes - old_holes
            holes_by_move[moves_played].append(new_holes)

            if lines_cleared in line_clear_counts:
                line_clear_counts[lines_cleared] += 1

            if hole_difference > 0:
                holes_created += hole_difference
            elif hole_difference < 0:
                holes_removed += abs(hole_difference)

            if block_class.__name__ == "BombBlock":

                bombs_used += 1

                bomb_cells_cleared += cleared_cells

                bomb_lines_cleared += lines_cleared

                height_reduction = max(
                    0,
                    old_height - new_height
                )

                hole_reduction = max(
                    0,
                    old_holes - new_holes
                )

                total_height_reduction += height_reduction
                total_hole_reduction += hole_reduction

            reward = agent.reward_function(
                new_board,
                lines_cleared,
                cleared_cells
            )

            total_score += 100 * (lines_cleared ** 2)

            board = new_board

            total_lines += lines_cleared
            moves_played += 1

        scores.append(total_score)
        lines_list.append(total_lines)
        moves_list.append(moves_played)
        tetris_list.append(tetrises)
        holes_created_list.append(holes_created)
        holes_removed_list.append(holes_removed)
        for k in total_line_clear_counts:
            total_line_clear_counts[k] += line_clear_counts[k]

        bombs_used_list.append(bombs_used)

        if bombs_used > 0:
            bomb_cells_list.append(
                bomb_cells_cleared / bombs_used
            )

            lines_after_bomb_list.append(
                bomb_lines_cleared / bombs_used
            )

            height_reduction_list.append(
                total_height_reduction / bombs_used
            )

            hole_reduction_list.append(
                total_hole_reduction / bombs_used
            )

    print("Evaluation over", num_games, "games")

    print("Average score:",
          mean(scores))

    print("Average lines cleared:",
          mean(lines_list))

    print("Average moves:",
          mean(moves_list))

    print("Average bombs used:",
          mean(bombs_used_list))
    if bomb_cells_list:
        print("Bomb efficiency (cells per bomb):",
        mean(bomb_cells_list))
    print("Average Tetrises:", mean(tetris_list))
    print("Average holes created:", mean(holes_created_list))
    print("Average holes removed:", mean(holes_removed_list))
    avg_holes = []

    for move in holes_by_move:
        if len(move) > 0:
            avg_holes.append(mean(move))
        else:
            avg_holes.append(None)
    labels = ["Single", "Double", "Triple", "Tetris"]
    values = [
        total_line_clear_counts[1],
        total_line_clear_counts[2],
        total_line_clear_counts[3],
        total_line_clear_counts[4]
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Number of occurrences")
    plt.title("Distribution of line clears with Bomb Piece (RL Agent)")
    plt.tight_layout()
    plt.savefig("line_clear_histogram_rl_bomb2.png", dpi=300)

    x = []
    y = []

    for i, value in enumerate(avg_holes):
        if value is not None:
            x.append(i + 1)
            y.append(value)

    plt.figure(figsize=(7,4))
    plt.plot(x, y)
    plt.xlabel("Move Number")
    plt.ylabel("Average Number of Holes")
    plt.title("Average Holes During Gameplay (RL Agent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("holes_over_time_rl_2.png", dpi=300)




    if bomb_cells_list:

        print(
            "Average cells cleared per bomb:",
            mean(bomb_cells_list)
        )

        print(
            "Average lines cleared after bomb:",
            mean(lines_after_bomb_list)
        )

        print(
            "Average height reduction after bomb:",
            mean(height_reduction_list)
        )

        print(
            "Average hole reduction after bomb:",
            mean(hole_reduction_list)
        )
        


if __name__ == "__main__":
    evaluate_agent()