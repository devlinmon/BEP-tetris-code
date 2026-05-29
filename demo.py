import time
import pygame

from board import Board
from game import game
from heuristic import HeuristicAgent
from rl_agent import RLAgent


CELL_SIZE = 30
ROWS = 15
COLS = 10
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE


COLORS = {
    0:  (20, 20, 20),      
    1: (255, 200, 46),     
    2: (254, 251, 52),
    3: (83, 218, 63),
    4: (1, 237, 250),
    5: (221, 10, 178),
    6: (234, 20, 28),
    7: (254, 72, 25),
    8: (0, 119, 211),     
}


def draw_board(screen, board, title="", score=0, lines=0, moves=0):
    screen.fill((0, 0, 0))

    font = pygame.font.SysFont(None, 28)

    title_text = font.render(title, True, (255, 255, 255))
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    lines_text = font.render(f"Lines: {lines}", True, (255, 255, 255))
    moves_text = font.render(f"Moves: {moves}", True, (255, 255, 255))

    screen.blit(title_text, (10, 5))
    screen.blit(score_text, (10, 30))
    screen.blit(lines_text, (10, 55))
    screen.blit(moves_text, (10, 80))

    y_offset = 110

    for r in range(ROWS):
        for c in range(COLS):
            value = board.grid[r][c]
            color = COLORS.get(value, (220, 40, 40))

            rect = pygame.Rect(
                c * CELL_SIZE,
                y_offset + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (80, 80, 80), rect, 1)

    pygame.display.flip()


def play_heuristic_demo(max_moves=500, delay=0.08):
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT + 110))
    pygame.display.set_caption("Heuristic Tetris Agent")

    g = game(use_bomb=True)
    agent = HeuristicAgent(g)
    board = Board()

    score = 0
    lines = 0
    moves = 0

    running = True

    while running and moves < max_moves:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        block = g.get_random_block()
        block_class = block.__class__

        move, _ = agent.best_move(board, block_class)

        if move is None:
            break

        rotation, col = move
        result = g.simulate_move(board, block_class, rotation, col)

        if result is None:
            break

        new_board, _, cleared_lines, _ = result

        board.grid = [row[:] for row in new_board.grid]

        score += 100 * (cleared_lines ** 2)
        lines += cleared_lines
        moves += 1

        draw_board(
            screen,
            board,
            title="Heuristic Agent",
            score=score,
            lines=lines,
            moves=moves,
        )

        time.sleep(delay)

    time.sleep(2)
    pygame.quit()


def play_rl_demo(
    model_path="final_tetris_rl_agent.pt",
    max_moves=500,
    delay=0.3,
):
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT + 110))
    pygame.display.set_caption("RL Tetris Agent")

    g = game(use_bomb=True)
    agent = RLAgent()
    agent.load(model_path)
    agent.epsilon = 0.0

    board = Board()

    score = 0
    lines = 0
    moves = 0

    running = True

    while running and moves < max_moves:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        block = g.get_random_block()
        block_class = block.__class__

        chosen = agent.select_action(board, block_class, training=False)

        if chosen is None:
            break

        rotation, col = chosen[0]

        result = g.simulate_move(board, block_class, rotation, col)

        if result is None:
            break

        new_board, _, cleared_lines, _ = result

        board.grid = [row[:] for row in new_board.grid]

        score += 100 * (cleared_lines ** 2)
        lines += cleared_lines
        moves += 1

        draw_board(
            screen,
            board,
            title="Reinforcement Learning Agent",
            score=score,
            lines=lines,
            moves=moves,
        )

        time.sleep(delay)

    time.sleep(2)
    pygame.quit()


if __name__ == "__main__":
    print("Choose agent:")
    
    choice = input("Enter choice: ")

    if choice == "1":
        play_heuristic_demo()
    elif choice == "2":
        play_rl_demo()
    else:
        print("Invalid choice")