from game import game
from board import Board
from blocks import Oblock, Iblock, Sblock, Zblock, Lblock, Jblock, Tblock

def test_simulate_move():
    g = game(None, None)
    b = Board()

    result = g.simulate_move(b, Oblock, 0, 4)

    assert result is not None

    new_board, placed_block, cleared_lines = result

    
    assert sum(cell != 0 for row in b.grid for cell in row) == 0

    
    assert sum(cell != 0 for row in new_board.grid for cell in row) == 4

    assert cleared_lines == 0
def test_board_init():
    b = Board()
    assert len(b.grid) == 20
    assert len(b.grid[0]) == 10

def test_block_positions():
    t = Tblock()
    assert len(t.get_position()) == 4

def test_place_block():
    b = Board()
    o = Oblock()
    b.place_block(o)
    filled = sum(cell != 0 for row in b.grid for cell in row)
    assert filled == 4
def test_clear_lines():
    b = Board()
    b.grid[19] = [1] * 10
    b.clear_lines()
    assert all(cell == 0 for cell in b.grid[19])
def drop_block(board, block):
    if not board.is_valid_position(block):
        return False
    while True:
        block.move(0, 1)
        if not board.is_valid_position(block):
            block.move(0, -1)
            break
    return True

def test_drop_empty_board():
    b = Board()
    o = Oblock()
    assert drop_block(b, o) == True
    ys = [tile.y for tile in o.get_position()]
    assert max(ys) == 19
    assert min(ys) == 18

def test_drop_on_obstacle():
    b = Board()
    b.grid[19][0] = 1
    b.grid[19][1] = 1
    o = Oblock()
    assert drop_block(b, o) == True
    ys = [tile.y for tile in o.get_position()]
    #xs = [tile.x for tile in o.get_position()]
    #print("positions:", list(zip(xs, ys)))
    #print("max y:", max(ys))
    assert max(ys) == 18

def test_blocked_spawn():
    b = Board()
    o = Oblock()
    for tile in o.get_position():
        b.grid[tile.y][tile.x] = 1
    assert drop_block(b, o) == False
def preview_block(board, block):
    temp = [row[:] for row in board.grid]
    for tile in block.get_position():
        temp[tile.y][tile.x] = block.id
    for row in temp:
        print(" ".join(str(cell) for cell in row))
    print()

def test_evaluate_board_empty():
    g = game(None, None)
    b = Board()

    score = g.evaluate_board(b, 0)

    assert score == 0

def test_best_move():
    g = game(None, None)
    b = Board()

    move, score = g.best_move(b, Oblock)

    assert move is not None
    assert isinstance(move, tuple)
    assert len(move) == 2

test_board_init()
test_block_positions()
test_place_block()
test_clear_lines()
test_drop_empty_board()
test_drop_on_obstacle()
test_blocked_spawn()
test_simulate_move()
test_evaluate_board_empty()
test_best_move()


print("All tests passed.")
