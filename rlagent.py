import numpy as np
from caro import Caro, Cell, Result
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.optimizers import Adam

inf = int(1e9)

class RLAgent:
    def __init__(self, game: Caro, cell: Cell, depth: int = 3):
        self.game = game
        self.cell = cell
        self.depth = depth
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.game.board.shape))
        model.add(Dense(32, activation='relu', input_shape=(0,1,2) + self.game.board.shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.game.board.size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model
    
    # Hàm train sẽ cập nhật giá trị của các ô trên bàn cờ dựa trên giá trị reward
    def train(self, old_state, action, reward, new_state, done):
        target = reward
        if not done:
            target += 0.95 * np.amax(self.model.predict(new_state.reshape((1, 1) + new_state.shape)))
        target_f = self.model.predict(old_state.reshape((1, 1) + old_state.shape))
        target_f[0][action] = target
        self.model.fit(old_state.reshape((1, 1) + old_state.shape), target_f, epochs=1, verbose=0)
        return target

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
        best_score = -inf
        best_move = None
        for row in range(self.game.size):
            for col in range(self.game.size):
                if self.game.board[row, col] == Cell.EMPTY:
                    self.game.move(row, col)
                    score = self.train(self.get_state(), self.get_action(self.get_state()), self.get_reward(self.game.check_win()), self.get_state(), self.game.check_win() != Result.PENDING)
                    self.game.unmove(row, col)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
        return best_move