from __future__ import annotations
from enum import IntEnum
import numpy as np

class Cell(IntEnum):
    EMPTY = 0
    X = 1
    O = 2

    def __str__(self):
        return ('_', 'X', 'O')[self]

class Result(IntEnum):
    PENDING = 0
    X_WIN = 1
    O_WIN = 2
    DRAW = 3

# X goes first
class Caro:
    def __init__(self, size: int, size_to_win: int, first_to_move: Cell = Cell.X):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.uint8)
        self.size_to_win = size_to_win
        self.turn = first_to_move
        self.status = Result.PENDING
        self.remaining_free_cells = size * size

    def clone(self) -> Caro:
        clone = Caro(self.size, self.size_to_win)
        clone.board = self.board.copy()
        clone.turn = self.turn
        clone.status = self.status
        clone.remaining_free_cells = self.remaining_free_cells
        return clone

    def move(self, row: int, col: int):
        if self.status != Result.PENDING or self.board[row, col] != Cell.EMPTY:
            raise ValueError("Invalid move")

        self.board[row, col] = self.turn
        self.remaining_free_cells -= 1

        if self.check_win_at(row, col):
            self.status = Result.X_WIN if self.turn == Cell.X else Result.O_WIN
        elif self.remaining_free_cells == 0:
            self.status = Result.DRAW

        self.turn = Cell.X if self.turn == Cell.O else Cell.O

    def unmmove(self, row: int, col: int):
        self.board[row, col] = Cell.EMPTY
        self.remaining_free_cells += 1
        self.status = Result.PENDING
        self.turn = Cell.X if self.turn == Cell.O else Cell.O

    def get_surroundings(self, offset=1) -> list[tuple[int, int]]:
        # get empty cells that are 2 cells away from nearest cell
        surroundings = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.remaining_free_cells > self.size * self.size // 2:
                    if self.board[row, col] != Cell.EMPTY:
                        for i in range(-offset, offset + 1):
                            for j in range(-offset, offset + 1):
                                if 0 <= row + i < self.size and 0 <= col + j < self.size and self.board[row + i, col + j] == Cell.EMPTY:
                                    surroundings.add((row + i, col + j))
                else:
                    if self.board[row, col] == Cell.EMPTY:
                        for i in range(-offset, offset + 1):
                            for j in range(-offset, offset + 1):
                                if 0 <= row + i < self.size and 0 <= col + j < self.size and self.board[row + i, col + j] != Cell.EMPTY:
                                    surroundings.add((row, col))
                                    break
        return list(surroundings)
    
    # get called after each move
    def check_win_at(self, row, col) -> bool:
        if self.board[row, col] == Cell.EMPTY:
            return False

        r1, r2 = self.get_row_consecutive(row, col)
        if r2 - r1 + 1 >= self.size_to_win:
            return True

        c1, c2 = self.get_col_consecutive(row, col)
        if c2 - c1 + 1 >= self.size_to_win:
            return True

        d1, d2 = self.get_main_diag_consecutive(row, col)
        if d2 + d1 + 1 >= self.size_to_win:
            return True

        ad1, ad2 = self.get_anti_diag_consecutive(row, col)
        if ad2 + ad1 + 1 >= self.size_to_win:
            return True

        return False

    def check_win(self) -> Result:
        return self.status

    # return [r1, r2] inclusive
    def get_row_consecutive(self, row, col) -> tuple[int, int]:
        cell = self.board[row, col]
        row_slice = self.board[row, :]
        start, end = col, col
        while start >= 0 and row_slice[start] == cell:
            start -= 1
        while end < self.size and row_slice[end] == cell:
            end += 1
        return start + 1, end - 1

    # return [c1, c2] inclusive
    def get_col_consecutive(self, row, col) -> tuple[int, int]:
        cell = self.board[row, col]
        col_slice = self.board[:, col]
        start, end = row, row
        while start >= 0 and col_slice[start] == cell:
            start -= 1
        while end < self.size and col_slice[end] == cell:
            end += 1
        return start + 1, end - 1

    # return [d1, d2] inclusive
    # you can get r1, c1 and r2, c2 by row - d1, col - d1 and row + d2, col + d2
    def get_main_diag_consecutive(self, row, col) -> tuple[int, int]:
        cell = self.board[row, col]
        diag_slice = self.board.diagonal(col - row)
        index = min(row, col)
        start, end = index, index
        while start >= 0 and diag_slice[start] == cell:
            start -= 1
        while end < len(diag_slice) and diag_slice[end] == cell:
            end += 1
        return index - (start + 1), end - 1 - index

    # return [d1, d2] inclusive
    # you can get r1, c1 and r2, c2 by row - d1, col + d1 and row - d2, col + d2
    def get_anti_diag_consecutive(self, row, col) -> tuple[int, int]:
        cell = self.board[row, col]
        anti_diag_slice = np.fliplr(self.board).diagonal((self.size - 1 - col) - row)
        index = min(row, self.size - 1 - col)
        start, end = index, index
        while start >= 0 and anti_diag_slice[start] == cell:
            start -= 1
        while end < len(anti_diag_slice) and anti_diag_slice[end] == cell:
            end += 1
        return index - (start + 1), end - 1 - index

    def random_free_cell(self) -> tuple[int, int]:
        free_cells = np.argwhere(self.board == Cell.EMPTY)
        return tuple(free_cells[np.random.choice(len(free_cells))])

    def __str__(self):
        return '\n'.join(' '.join(('_', 'X', 'O')[cell] for cell in row) for row in self.board)
