import numpy as np

class StateActionFeatureVectorWithTile:
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width) + 1
        self.num_features = int(self.num_actions * self.num_tilings * np.prod(self.num_tiles))

    def feature_vector_len(self) -> int:
        return self.num_features

    def __call__(self, s, done, a) -> np.array:
        if done:
            return np.zeros(self.feature_vector_len())
        else:
            feature_vector = np.zeros(self.feature_vector_len())
            state_tiles = self.get_state_tiles(s)
            action_offset = a * self.num_tilings * np.prod(self.num_tiles)
            for tile in state_tiles:
                feature_vector[int(action_offset + tile)] = 1
            return feature_vector.flatten()

    def get_state_tiles(self, s):
        """
        Calculate the tiles for the given state s across all tilings.
        """
        # Extract the state array from the tuple
        s = s[0]
        s = np.array(s)
        state_diff = s - self.state_low
        tiling_indices = np.arange(self.num_tilings)
        offsets = (self.tile_width / self.num_tilings) * tiling_indices[:, np.newaxis]
        state_tiles = np.floor((state_diff + offsets) / self.tile_width).astype(int)

        # Clip state_tiles to ensure values are within valid ranges
        state_tiles = np.clip(state_tiles, 0, self.num_tiles.astype(int) - 1)

        try:
            ravel_indices = np.ravel_multi_index(state_tiles.T, self.num_tiles.astype(int))
            return ravel_indices
        except ValueError as e:
            print("Error in ravel_multi_index:", e)
            raise

def SarsaLambda(
    env,  # openai gym environment
    gamma: float,  # discount factor
    lam: float,  # decay rate
    alpha: float,  # step size
    X: StateActionFeatureVectorWithTile,
    num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(λ) based on the provided pseudocode.
    """
    def epsilon_greedy_policy(s, done, w, epsilon=0.2):  # Increase epsilon for more exploration
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        return np.argmax(Q) if np.random.rand() >= epsilon else np.random.randint(nA)


    w = np.random.randn(X.feature_vector_len()) * 0.01  # Initialize w with small random values


    for episode in range(num_episode):
        s = env.reset()
        G = 0  # Initialize total reward for this episode
        done = False
        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0

        while not done:
            outcome = env.step(a)
            if len(outcome) == 5:
                s_prime, r, done, truncated, _ = outcome
            else:
                s_prime, r, done, _ = outcome
                truncated = False

            a_prime = epsilon_greedy_policy(s_prime, done, w)
            x_prime = X(s_prime, done, a_prime)

            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)

            delta = r + gamma * Q_prime - Q

            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x

            Q_old = Q
            x = x_prime
            a = a_prime
            G += r

            print(f"State: {s}, Action: {a}, Reward: {r}, Q-value: {Q}, Next Q-value: {Q_prime}, Total Reward: {G}")

            if done or truncated:
                break

        print(f"Episode {episode}: Total Reward: {G}")


        return w
