class Board:
    def __init__(self):
        self.nr_rows = 20
        self.nr_cols = 10
        self.grid = [[0 for j in range(self.nr_cols)] for i in range(self.nr_rows)]

    def print_grid(self):     
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))

    def is_valid_position(self, block):
        for tile in block.get_position():
            if tile.x < 0 or tile.x >= self.nr_cols:
                return False
            if tile.y < 0 or tile.y >= self.nr_rows:
                return False
            if self.grid[tile.y][tile.x] != 0:
                return False
        return True
    def place_block(self, block):
        for tile in block.get_position():
            self.grid[tile.y][tile.x] = block.id

    def clear_lines(self):
        new_grid = [row for row in self.grid if 0 in row]
        cleared = self.nr_rows - len(new_grid)

        for i in range(cleared):
            new_grid.insert(0, [0] * self.nr_cols)

        self.grid = new_grid
        return cleared
    def clear_area(self, center_x, center_y, radius=1):
        cleared_cells = 0

        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):

                if 0 <= y < self.nr_rows and 0 <= x < self.nr_cols:

                    if self.grid[y][x] != 0:
                        cleared_cells += 1

                    self.grid[y][x] = 0

        return cleared_cells

 