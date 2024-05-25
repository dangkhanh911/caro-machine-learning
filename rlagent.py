import numpy as np
from caro import Caro, Cell
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.optimizers import Adam

class RLAgent:
    def __init__(self, game: Caro, cell: Cell, depth: int = 3):
        self.game = game
        self.cell = cell
        self.depth = depth
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.game.board.shape))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.game.board.size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def get_state(self):
        return np.copy(self.game.board)

    def get_action(self, state):
        return np.argmax(self.model.predict(state.reshape((1, 1) + state.shape)))

    def train(self, old_state, action, reward, new_state, done):
        target = reward
        if not done:
            target += 0.95 * np.amax(self.model.predict(new_state.reshape((1, 1) + new_state.shape)))
        target_f = self.model.predict(old_state.reshape((1, 1) + old_state.shape))
        target_f[0][action] = target
        self.model.fit(old_state.reshape((1, 1) + old_state.shape), target_f, epochs=1, verbose=0)

    def generate_optimal_move(self):
        state = self.get_state()
        best_action = self.get_action(state)
        return best_action // self.game.size, best_action % self.game.size
    