from game import game
from board import Board
from heuristic import HeuristicAgent

def run_experiment(num_games=500, max_moves=1000):
    g = game(use_bomb=True)
    agent = HeuristicAgent(g)

    total_moves = 0
    total_lines = 0
    total_score = 0
    total_bombs_used = 0
    total_bomb_cells = 0

    for _ in range(num_games):
        board = Board()
        moves = 0
        lines = 0
        score = 0
        bombs_used = 0
        bomb_cells_cleared = 0  

        while moves < max_moves:
            block = g.get_random_block()
            block_class = block.__class__

            move, _ = agent.best_move(board, block_class)

            if move is None:
                break

            rotation, col = move
            result = g.simulate_move(board, block_class, rotation, col)

            if result is None:
                break

            new_board, placed_block, cleared_lines, cleared_cells = result
            if block_class.__name__ == "BombBlock":
                bombs_used += 1
                bomb_cells_cleared += cleared_cells


            board.grid = new_board.grid

            lines += cleared_lines
            score += 100 * (cleared_lines ** 2)
            moves += 1

        total_moves += moves
        total_lines += lines
        total_score += score
        total_bombs_used += bombs_used
        total_bomb_cells += bomb_cells_cleared

    print("Average moves:", total_moves / num_games)
    print("Average lines:", total_lines / num_games)
    print("Average score:", total_score / num_games)
    print("Average bombs used:", total_bombs_used / num_games)
    if total_bombs_used > 0:
        print("Average cells cleared per bomb:", total_bomb_cells / total_bombs_used)


if __name__ == "__main__":
    run_experiment()