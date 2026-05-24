from position import position
from board import Board
class block:
    def __init__(self, id):
        self.id = id
        self.cells = {}
        self.rotation = 0
        self.offset_x = 0
        self.offset_y = 0
    def move(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        
    def get_position(self):
        tiles=self.cells[self.rotation]
        moved_tiles = []
        for tile in tiles:
            tile=position(tile.x + self.offset_x, tile.y + self.offset_y)
            moved_tiles.append(tile)
        return moved_tiles
