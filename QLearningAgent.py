import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, gamma, alpha, epsilon, epsilon_decay, epsilon_min):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # a dictionary that maps tuples -> action values
        # if a state is new 0s are returned
        self.q_table = []

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def choose_action(self, state_key):
        # Exploration greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.n_actions)

        # gets the best known value (exploitation)
        q_values = self.get_q_values(state_key)
        # randomly choose among the best if there is a tie (prevents from getting stuck)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)