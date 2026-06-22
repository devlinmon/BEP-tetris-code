import time
import pygame

from board import Board
from game import game
from heuristic import HeuristicAgent
from rl_agent import RLAgent


cell_size = 30
rows = 20
cols = 10
width = cols * cell_size
height = rows * cell_size


Colours = {
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

    for r in range(rows):
        for c in range(cols):
            value = board.grid[r][c]
            color = Colours.get(value, (220, 40, 40))

            rect = pygame.Rect(
                c * cell_size,
                y_offset + r * cell_size,
                cell_size,
                cell_size,
            )

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (80, 80, 80), rect, 1)

    pygame.display.flip()


def play_heuristic_demo(max_moves=500, delay=0.05):
    pygame.init()

    screen = pygame.display.set_mode((width, height + 110))
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
            pygame.image.save(screen, "heuristic_final_board.png")
            break

        rotation, col = move
        result = g.simulate_move(board, block_class, rotation, col)

        if result is None:
            pygame.image.save(screen, "heuristic_final_board.png")
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
    model_path="tetris_rl_state_high.pt",
    max_moves=500,
    delay=0.01,
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
            pygame.image.save(screen, "rl_final_board.png")
            break

        rotation, col = chosen[0]

        result = g.simulate_move(board, block_class, rotation, col)

        if result is None:
            pygame.image.save(screen, "rl_final_board.png")
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