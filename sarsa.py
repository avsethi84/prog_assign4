# In this part, we implement the true online Sarsa() algorithm as described on page 307 of the textbook. Open `sarsa.py` and implement `SarsaLambda` method. Tile coding should be used as a feature extractor . To do so, you should fill in the parts indicated as 'TODO' in the `StateActionFeatureVectorWithTile` class.

# Here are a few tips & specific details you need to be careful when implementing tile coding:

# - Each tiling should cover the entire space completely so that every point is included by the same number of tilings.
# - In order to achieve that, # tiles in each dimension should be: ceil[(high-low)/tile_width] + 1.
# - Each tiling should start from (low - tiling_index / # tilings * tile width) where the tiling index starts from 0. With the starting offset, you can easily find out the corresponding tile for each tiling given the state.

# Note: It's possible to have the wrong number of tilings in test_sarsa.py due to floating point errors that cause your implementation to use one tile more or less. If that happens, try slightly increasing the tile width specified in test_sarsa.py by changing line 15 to

# tile_width=np.array([.451,.0351]))


import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width) + 1
        self.num_features = self.num_actions * self.num_tilings * np.prod(self.num_tiles)


    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_features
    

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(X.feature_vector_len())
        else:
            feature_vector = np.zeros(X.feature_vector_len())
            state_tiles = get_state_tiles(s, X)
            action_offset = a * X.num_tilings * np.prod(X.num_tiles)
            for tile in state_tiles:
                feature_vector[action_offset + tile] = 1
            return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for episode in range(num_episode):
        s = env.reset()
        done = False
        a = epsilon_greedy_policy(s, done, w)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0
        
        while not done:
            s_prime, r, done, _ = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime, done, w)
            delta = r + gamma * np.dot(w, X(s_prime, done, a_prime)) - np.dot(w, X(s, done, a))
            z = gamma * lam * z + X(s, done, a)
            w += alpha * (delta + np.dot(w, X(s, done, a)) - Q_old) * z - alpha * (np.dot(w, X(s, done, a)) - Q_old) * X(s, done, a)
            Q_old = np.dot(w, X(s, done, a))
            s = s_prime
            a = a_prime

    return w
