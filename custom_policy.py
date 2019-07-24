import random
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K
from policies import base_policy as bp
import numpy as np

EPSILON = 0.8
NUM_FEATURES = 1
ACTIONS_IDX = {'L': 0, 'R': 1, 'F': 2}
NUM_ACTIONS = 3
BATCH_SIZE = 5

class Custom204022487(bp.Policy):
    """
    A policy that base on Deep Q learning.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.learn_rate = 0.001
        self.decay_rate = 0.7
        self.tmp_epsilon = EPSILON
        self.history = deque(maxlen=1000)
        self.model = self._init_model()

    def _init_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=66, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learn_rate))
        return model

    def _state2features(self, state):
        f = np.zeros(66)
        section = 0

        board, head = state
        head_pos, direction = head

        for act in bp.Policy.ACTIONS:
            new_pos = head_pos.move(bp.Policy.TURNS[direction][act])
            new_dir = bp.Policy.TURNS[direction][act]
            r = new_pos[0]
            c = new_pos[1]
            f_idx = board[r, c] + 1
            f[(section * 11) + f_idx] = 1
            section += 1

            next_position = new_pos.move(bp.Policy.TURNS[new_dir]['F'])
            r = next_position[0]
            c = next_position[1]
            f_idx = board[r, c] + 1
            f[(section * 11) + f_idx] = 1
            section += 1
        return np.array([f])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if len(self.history) < BATCH_SIZE:
            return
        minibatch = random.sample(self.history, BATCH_SIZE)
        for next_state, reward, state, action in minibatch:
            act_idx = ACTIONS_IDX[action]
            target = reward + self.decay_rate * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][act_idx] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        try:
            if round < self.game_duration - self.score_scope:
                if round % 100 == 0:
                    self.epsilon = max(0.05, self.tmp_epsilon * 0.99)
                    self.tmp_epsilon = self.epsilon
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward
                if round % 1000 == 0:
                    self.epsilon = 0.005

            elif round > self.game_duration - self.score_scope:
                # self.epsilon = 0.05
                self.epsilon = 0

                if round % 100 == 0:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        f = self._state2features(new_state)
        if round > 2:
            self.history.append((f, reward, self._state2features(prev_state), prev_action))

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            action_val = self.model.predict(f)
            act_idx = np.argmax(action_val[0])
            if act_idx == np.argmin(action_val[0]):
                return bp.Policy.DEFAULT_ACTION
            else:
                return bp.Policy.ACTIONS[act_idx]


