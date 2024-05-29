import numpy as np
import random
from caro import Caro, Cell, Result
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.optimizers import Adam

inf = int(1e9)

class RLAgent:
    def __init__(self, game: Caro, cell: Cell):
        self.game = game
        self.cell = cell
        self.model = self.build_model()
        self.memory = []  # Step 1: Create a memory
        self.batch_size = 32  # Size of the batches to train on

    # count how many consecutive cells
    # this is neutral to the player
    # meaning that it will return the same value for both player
    def evaluate_at(self, row, col, visited: np.ndarray) -> int:
        if visited[row, col]:
            return 0
        visited[row, col] = 1
        score = 0

        # row
        r1, r2 = self.game.get_row_consecutive(row, col)
        is_blocked_left = r1 == 0 or self.game.board[row, r1 - 1] != Cell.EMPTY
        is_blocked_right = r2 == self.game.size - 1 or self.game.board[row, r2 + 1] != Cell.EMPTY
        # print("Row consecutive cells:", r2 - r1 + 1)
        score += (r2 - r1 + 1) ** 2 * (2 - is_blocked_left - is_blocked_right)
        visited[row, r1:r2 + 1] = 1

        # col
        c1, c2 = self.game.get_col_consecutive(row, col)
        is_blocked_up = c1 == 0 or self.game.board[c1 - 1, col] != Cell.EMPTY
        is_blocked_down = c2 == self.game.size - 1 or self.game.board[c2 + 1, col] != Cell.EMPTY
        # print("Column consecutive cells:", c2 - c1 + 1)
        score += (c2 - c1 + 1) ** 2 * (2 - is_blocked_up - is_blocked_down)
        visited[c1:c2 + 1, col] = 1

        # main diag
        d1, d2 = self.game.get_main_diag_consecutive(row, col)
        r1, c1 = row - d1, col - d1
        r2, c2 = row + d2, col + d2
        is_blocked_up_left = (r1 == 0 or c1 == 0) or self.game.board[r1 - 1, c1 - 1] != Cell.EMPTY
        is_blocked_down_right = (r2 == self.game.size - 1 or c2 == self.game.size - 1) or self.game.board[r2 + 1, c2 + 1] != Cell.EMPTY
        # print("Main diagonal consecutive cells:", d2 - d1 + 1)
        score += (d2 + d1 + 1) ** 2 * (2 - is_blocked_up_left - is_blocked_down_right)
        for i in range(-d1, d2 + 1):
            visited[row + i, col + i] = 1

        # anti diag
        ad1, ad2 = self.game.get_anti_diag_consecutive(row, col)
        r1, c1 = row - ad1, col + ad1
        r2, c2 = row + ad2, col - ad2
        is_blocked_up_right = (r1 == 0 or c1 == self.game.size - 1) or self.game.board[r1 - 1, c1 + 1] != Cell.EMPTY
        is_blocked_down_left = (r2 == self.game.size - 1 or c2 == 0) or self.game.board[r2 + 1, c2 - 1] != Cell.EMPTY
        # print("Anti diagonal consecutive celss:", ad2 - ad1 + 1)
        score += (ad2 + ad1 + 1) ** 2 * (2 - is_blocked_up_right - is_blocked_down_left)
        for i in range(-ad1, ad2 + 1):
            visited[row + i, col - i] = 1

        # print("Score:", score)
        return score

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.game.board.shape))
        model.add(Dense(32, activation='relu', input_shape=(0,1,2) + self.game.board.shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.game.board.size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
    def remember(self, old_state, action, reward, new_state, done):
        self.memory.append((old_state, action, reward, new_state, done))  # Step 2: Store experiences

    def train(self) -> tuple[int, tuple[int, int] | None]:
        if len(self.memory) < self.batch_size:
            return 0, None
        old_state, action, reward, new_state, done = random.choice(self.memory)  # Step 3: Sample a random experience
        target = reward
        if not done:
            target = reward + self.model.predict(new_state.reshape((1, 1) + new_state.shape))[0][action]
        self.model.fit(old_state.reshape((1, 1) + old_state.shape), target, epochs=1, verbose=0)  # Step 4: Train the model
        return action, None if done else self.get_action(new_state)


    # Hàm get_state sẽ trả về trạng thái hiện tại của bàn cờ
    def get_state(self):
        return np.copy(self.game.board)
    
    # Hàm get_action sẽ trả về nước đi tốt nhất dựa trên giá trị reward
    def get_action(self, state):
        return np.argmax(self.model.predict(state.reshape((1, 1) + state.shape)))
    
    # Hàm get_reward sẽ trả về giá trị reward dựa trên kết quả của trò chơi
    def get_reward(self, result):
        if result == Result.DRAW:
            return 0
        if result == Result.X_WIN and self.cell == Cell.X:
            return 1
        if result == Result.O_WIN and self.cell == Cell.O:
            return 1
        return -1

    def generate_random_move(self) -> tuple[int, int]:
        empty_cells = np.argwhere(self.game.board == Cell.EMPTY)
        return empty_cells[np.random.randint(len(empty_cells))]

    def generate_optimal_move(self) -> tuple[int, int]:
        _, move = self.train()
        if move is None:
            return self.generate_random_move()
        return move