import random
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K
from policies import base_policy as bp
import numpy as np

EPSILON = 0.9
NUM_FEATURES = 4
ACTIONS_IDX = {'L': 0, 'R': 1, 'F': 2}
NUM_ACTIONS = 3
BATCH_SIZE = 5

class Custom204022487(bp.Policy):
    """
    A policy base on linear Q learning by using NN.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.learn_rate = 0.001
        self.decay_rate = 0.7
        self.history = deque(maxlen=1000)
        self.model = self._init_model()

    def _init_model(self):
        model = Sequential()
        model.add(Dense(4, input_dim=44, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learn_rate))
        return model

    def _state2features(self, state, act):
        f = np.zeros(44)
        section = 0
        board, head = state
        head_pos, direction = head

        new_pos = head_pos.move(bp.Policy.TURNS[direction][act])
        new_dir = bp.Policy.TURNS[direction][act]
        r = new_pos[0]
        c = new_pos[1]
        f_idx = board[r, c] + 1  # shift the values to 0-10
        f[f_idx + (11 * section)] = 1

        for a in bp.Policy.ACTIONS:
            section += 1
            next_position = new_pos.move(bp.Policy.TURNS[new_dir][a])
            r = next_position[0]
            c = next_position[1]
            f_idx = board[r, c] + 1
            f[f_idx + (11 * section)] = 1

        return np.array([f])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if len(self.history) < BATCH_SIZE:
            return
        minibatch = random.sample(self.history, BATCH_SIZE)
        for f, next_q_val, reward in minibatch:
            target = reward + self.decay_rate * next_q_val
            target_f = self.model.predict(f)
            target_f[0] = target
            self.model.fit(f, target_f, epochs=1, verbose=0)

        try:
            if round < self.game_duration - self.score_scope:
                if round % 100 == 0:
                    self.epsilon = max(0.005, self.epsilon*0.99)
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward

            if round > self.game_duration - self.score_scope:
                self.epsilon = 0.005
                if round % 100 == 0:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def _get_next_action(self, state):
        act_val = []
        for a in bp.Policy.ACTIONS:
            f = self._state2features(state, a)
            q_val = self.model.predict(f)[0]
            act_val.append(q_val)
        return bp.Policy.ACTIONS[np.argmax(act_val)], np.amax(act_val)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 2:
            f = self._state2features(prev_state, prev_action)
            next_act, next_q = self._get_next_action(new_state)
            self.history.append((f, next_q, reward))

            if np.random.rand() < self.epsilon:
                return np.random.choice(bp.Policy.ACTIONS)
            else:
                return next_act
        else:
            return np.random.choice(bp.Policy.ACTIONS)




