from policies import base_policy as bp
import numpy as np

EPSILON = 0.8
NUM_VALUES = 11
NUM_FEATURES = 44

class Linear204022487(bp.Policy):
    """
    A policy that base on linear Q learning method.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.q_weights = np.random.random(NUM_FEATURES + 1)
        self.learn_rate = 0.3
        self.decay_rate = 0.5

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        f = self._state2features(prev_state, prev_action)
        q_value = np.dot(self.q_weights, f)
        next_q_value = np.dot(self.q_weights, self._state2features(new_state, self._get_action(new_state)))

        loss = reward + (self.decay_rate * next_q_value) - q_value
        self.q_weights += self.learn_rate * loss * f

        try:
            if round < self.game_duration - self.score_scope:
                if round % 100 == 0:
                    self.epsilon -= 0.005
                    self.learn_rate -= 0.005
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward

            if round > self.game_duration - self.score_scope:
                # self.epsilon = 0.05
                self.epsilon = 0
                # self.learn_rate = 0.001
                if round % 100 == 0:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                    self.r_sum = 0
                else:
                    self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def _state2features(self, state, act):
        '''
        Get state and action and returns representation of the next state and the next options around it.
        '''
        f = np.zeros(NUM_FEATURES + 1)
        section = 0
        board, head = state
        head_pos, direction = head

        new_pos = head_pos.move(bp.Policy.TURNS[direction][act])
        new_dir = bp.Policy.TURNS[direction][act]
        r = new_pos[0]
        c = new_pos[1]
        f_idx = board[r, c] + 1  # shift the values to 0-10
        f[f_idx + (NUM_VALUES * section)] = 1

        for a in bp.Policy.ACTIONS:
            section += 1
            next_position = new_pos.move(bp.Policy.TURNS[new_dir][a])
            r = next_position[0]
            c = next_position[1]
            f_idx = board[r, c] + 1
            f[f_idx + (NUM_VALUES * section)] = 1
        f[-1] = 1
        return f

    def _get_action(self, new_state):
        '''
        Get state and returns the optimal action according to the current model.
        '''
        actions_val = []
        for a in bp.Policy.ACTIONS:
            f = self._state2features(new_state, a)
            actions_val.append(np.dot(self.q_weights, f))
        if np.argmax(actions_val) == np.argmin(actions_val):
            return bp.Policy.DEFAULT_ACTION
        else:
            return bp.Policy.ACTIONS[np.argmax(actions_val)]

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            return self._get_action(new_state)





