import numpy as np
from caro import Caro, Cell, Result

class Human:
    def __init__(self, game: Caro, cell: Cell):
        self.game = game
        self.cell = cell

    def generate_optimal_move(self):
        while True:
            try:
                row, col = map(int, input('Enter your move (row, col): ').split())
                if 0 <= row < self.game.size and 0 <= col < self.game.size and self.game.board[row, col] == Cell.EMPTY:
                    return row, col
                else:
                    print('Invalid move')
            except ValueError:
                print('Invalid input')