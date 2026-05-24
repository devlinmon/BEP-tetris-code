from board import Board

grid = Board()
grid.print_grid()

from board import Board
from blocks import Tblock

b = Board()
t = Tblock()

print(b.is_valid_position(t))

from board import Board
from blocks import Oblock

b = Board()
o = Oblock(3)

print(b.is_valid_position(o))   # expected: True

o.move(3, 0)
print(b.is_valid_position(o))   # expected: False

b = Board()
o = Oblock(3)
b.place_block(o)
b.print_grid()

filled = sum(cell != 0 for row in b.grid for cell in row)
print(filled)   # should be 4
from board import Board
b = Board()
b.grid[19] = [1] * 9
b.clear_lines()
b.print_grid()

from board import Board
from blocks import Oblock

def drop_block(board, block):
    while True:
        block.move(0, 1)
        if not board.is_valid_position(block):
            block.move(0, -1)
            break
from board import Board
from blocks import Oblock

b = Board()
b.grid[19][4] = 9
b.grid[19][5] = 9

o = Oblock(2)
drop_block(b, o)

for tile in o.get_position():
    print(tile.x, tile.y)

b = Board()
o = Oblock(2)
drop_block(b, o)

for tile in o.get_position():
    print(tile.x, tile.y)


from game import game
from board import Board
from blocks import Tblock

g = game(None, None)
b = Board()

move, score = g.best_move(b, Tblock)

print("Best move:", move)
print("Best score:", score)

from game import game

g = game(None, None)
moves, lines, final_board, score = g.simulate_game(max_moves=1000, print_board=True)

print("Game over")
print("Moves played:", moves)
print("Total lines cleared:", lines)
print("Final score:", score)
print("Final board state:")
final_board.print_grid()

from game import game

num_games = 100
total_moves = 0
total_lines = 0

for i in range(num_games):
    g = game()
    moves, lines, board, score = g.simulate_game(max_moves=1000, print_board=True)

    total_moves += moves
    total_lines += lines

print("Average moves:", total_moves / num_games)
print("Average lines cleared:", total_lines / num_games)

from game import game
from board import Board
from heuristic import HeuristicAgent

g = game()
agent = HeuristicAgent(g)

board = Board()

while True:
    block = g.get_random_block()
    block_class = block.__class__

    move, score = agent.best_move(board, block_class)

    if move is None:
        break

    rotation, col = move
    result = g.simulate_move(board, block_class, rotation, col)

    if result is None:
        break

    new_board, _, cleared_lines = result
    board.grid = new_board.grid