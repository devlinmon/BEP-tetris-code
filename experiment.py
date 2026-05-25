from statistics import mean
from game import game
from board import Board
from heuristic import HeuristicAgent


def run_experiment(num_games=500, max_moves=1000):

    g = game(use_bomb=True)
    agent = HeuristicAgent(g)

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

        moves = 0
        lines = 0
        score = 0

        bombs_used = 0
        bomb_cells_cleared = 0
        bomb_lines_cleared = 0

        total_height_reduction = 0
        total_hole_reduction = 0

        while moves < max_moves:

            block = g.get_random_block()
            block_class = block.__class__

            move, _ = agent.best_move(board, block_class)

            if move is None:
                break

            rotation, col = move

            old_heights = g.get_column_heights(board)
            old_height = sum(old_heights)

            old_holes = g.count_holes(board)

            result = g.simulate_move(
                board,
                block_class,
                rotation,
                col
            )

            if result is None:
                break

            new_board, placed_block, cleared_lines, cleared_cells = result

            new_heights = g.get_column_heights(new_board)
            new_height = sum(new_heights)

            new_holes = g.count_holes(new_board)

            if block_class.__name__ == "BombBlock":

                bombs_used += 1

                bomb_cells_cleared += cleared_cells
                bomb_lines_cleared += cleared_lines

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

            board.grid = [row[:] for row in new_board.grid]

            lines += cleared_lines
            score += 100 * (cleared_lines ** 2)
            moves += 1

        scores.append(score)
        lines_list.append(lines)
        moves_list.append(moves)

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

    print("Average lines:",
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
    run_experiment()