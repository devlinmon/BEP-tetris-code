from statistics import mean
from board import Board
from game import game
from rl_agent import RLAgent


def evaluate_agent(
    model_path: str = "final_tetris_rl_agent.pt",
    num_games: int = 100,
    max_moves: int = 1000,
):
    sim = game()
    agent = RLAgent()
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

    for _ in range(num_games):

        board = Board()

        total_score = 0.0
        total_lines = 0
        moves_played = 0

        bombs_used = 0
        bomb_cells_cleared = 0

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

            board.grid = [row[:] for row in new_board.grid]

            total_lines += lines_cleared
            moves_played += 1

        scores.append(total_score)
        lines_list.append(total_lines)
        moves_list.append(moves_played)

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