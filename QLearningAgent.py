import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, gamma=0.1, alpha=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # a dictionary that maps tuples -> action values
        # if a state is new 0s are returned
        self.q_table = {}

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def choose_action(self, state_key):
        # Exploration greedy
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.randint(0,self.n_actions-1)

        # gets the best known value (exploitation)
        q_values = self.get_q_values(state_key)
        # randomly choose among the best if there is a tie (prevents from getting stuck)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        # get current q value
        q_values = self.get_q_values(state)
        current_q = q_values[action]

        # get max future Q-value
        next_q_values = self.get_q_values(next_state)
        max_next_q = np.max(next_q_values)

        # Bellman Equation
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay