class HeuristicAgent:
    def __init__(self, game,
                 w_lines=10.0,
                 w_height=-0.5,
                 w_holes=-2.0,
                 w_bumpiness=-0.5):
        self.game = game


        self.w_lines = w_lines
        self.w_height = w_height
        self.w_holes = w_holes
        self.w_bumpiness = w_bumpiness

    def evaluate_board(self, board, lines_cleared):
        heights = self.game.get_column_heights(board)
        aggregate_height = sum(heights)
        holes = self.game.count_holes(board)
        bumpiness = self.game.get_bumpiness(heights)

        score = (
            self.w_lines * (lines_cleared ** 2)
            + self.w_height * aggregate_height
            + self.w_holes * holes
            + self.w_bumpiness * bumpiness
        )

        return score

    def best_move(self, board, block_class):
        best_score = float("-inf")
        best_move = None

        for rotation in block_class().cells.keys():
            for col in range(board.nr_cols):
                result = self.game.simulate_move(board, block_class, rotation, col)

                if result is None:
                    continue

                new_board, _, cleared_lines, _ = result
                score = self.evaluate_board(new_board, cleared_lines)

                if score > best_score:
                    best_score = score
                    best_move = (rotation, col)

        return best_move, best_score