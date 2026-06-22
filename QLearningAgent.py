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