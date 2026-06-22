from board import Board
from blocks import Oblock, Iblock, Sblock, Zblock, Lblock, Jblock, Tblock, BombBlock
import random
import copy


class game:
    def __init__(self, board=None, blocks=None, use_bomb=False, bomb_probability=0.03):
        self.grid = Board()
        self.blocks = [Oblock, Iblock, Sblock, Zblock, Lblock, Jblock, Tblock]
        self.use_bomb = use_bomb
        self.bomb_probability = bomb_probability
    def get_random_block(self):
        if self.use_bomb and random.random() < self.bomb_probability:
            return BombBlock()

        block_class = random.choice(self.blocks)
        return block_class()
    def drop_block(self, board, block):
        if not board.is_valid_position(block):
            return False

        while True:
            block.move(0, 1)
            if not board.is_valid_position(block):
                block.move(0, -1)
                break

        return True
    def game_over(self, board, block):
        return not board.is_valid_position(block)
    
    def get_valid_columns_for_rotation(self, board, block_class, rotation):
        
        block = block_class()
        cells = block.cells[rotation]

        min_x = min(cell.x for cell in cells)
        max_x = max(cell.x for cell in cells)

        start_col = -min_x
        end_col = board.nr_cols - max_x

        return range(start_col, end_col)


    def get_legal_placements(self, board, block_class):
        placements = []

        for rotation in block_class().cells.keys():
            for col in self.get_valid_columns_for_rotation(board, block_class, rotation):
                test_block = block_class()
                test_block.rotation = rotation
                test_block.offset_x = col
                test_block.offset_y = 0

                if board.is_valid_position(test_block):
                    placements.append((rotation, col))

        return placements
    def simulate_move(self, board, block_class, rotation, col):
        new_board = Board()
        new_board.grid = [row[:] for row in board.grid]

        test_block = block_class()
        test_block.rotation = rotation
        test_block.offset_x = col
        test_block.offset_y = 0

        if not new_board.is_valid_position(test_block):
            return None

        dropped = self.drop_block(new_board, test_block)
        if not dropped:
            return None

        
        if isinstance(test_block, BombBlock):

            bomb_tile = test_block.get_position()[0]

            cleared_cells = new_board.clear_area(
                bomb_tile.x,
                bomb_tile.y,
                radius=1
            )

            cleared_lines = new_board.clear_lines()

            return new_board, test_block, cleared_lines, cleared_cells

        new_board.place_block(test_block)
        cleared_lines = new_board.clear_lines()

        return new_board, test_block, cleared_lines, 0
    
    def get_column_heights(self, board):
        heights = []

        for col in range(board.nr_cols):
            height = 0
            for row in range(board.nr_rows):
                if board.grid[row][col] != 0:
                    height = board.nr_rows - row
                    break
            heights.append(height)

        return heights
    
    def count_holes(self, board):
        holes = 0

        for col in range(board.nr_cols):
            block_found = False
            for row in range(board.nr_rows):
                if board.grid[row][col] != 0:
                    block_found = True
                elif block_found and board.grid[row][col] == 0:
                    holes += 1

        return holes

    def get_bumpiness(self, heights):
        bumpiness = 0

        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        return bumpiness

    def evaluate_board(self, board, lines_cleared):
        heights = self.get_column_heights(board)
        aggregate_height = sum(heights)
        holes = self.count_holes(board)
        bumpiness = self.get_bumpiness(heights)

        score = (
            10 * (lines_cleared**2)
            - 0.5 * aggregate_height
            - 2 * holes
            - 0.5 * bumpiness
        )

        return score
    
    def best_move(self, board, block_class):
        best_score = float("-inf")
        best_move = None

        placements = self.get_legal_placements(board, block_class)

        for rotation, col in placements:
            result = self.simulate_move(board, block_class, rotation, col)
            if result is None:
                continue

            new_board, _, cleared_lines = result
            score = self.evaluate_board(new_board, cleared_lines)

            if score > best_score:
                best_score = score
                best_move = (rotation, col)

        return best_move, best_score

        return best_move, best_score
    def play(self,board, block_class):
        move, score = self.best_move(board, block_class)
        if move is None:
            return False, 0

        rotation, col = move

        result = self.simulate_move(board, block_class, rotation, col)

        if result is None:
            return False, 0

        new_board, placed_block, cleared_lines = result
        board.grid = new_board.grid

        return True, cleared_lines
    
    

    def simulate_game(self, max_moves, print_board=False):
        board = Board()
        total_lines_cleared = 0
        moves_played = 0
        score=0

        while moves_played < max_moves:
            block = self.get_random_block()
            block_class = block.__class__

            success, cleared_lines = self.play(board, block_class)

            if not success:
                break

            total_lines_cleared += cleared_lines
            score = (100 * (cleared_lines ** 2)) + score
            moves_played += 1

        return moves_played, total_lines_cleared, board, score