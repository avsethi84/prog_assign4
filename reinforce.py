# Now, we will implement the REINFORCE algorithm with baseline. To do so, we model the policy and the value function for a baseline with neural networks. We again highly suggest you to use Pytorch (and not Tensorflow) as the deep learning framework. Ignore the comments in the code for Tensorflow as they were for a previous iteration of the course. If you are not familiar with Pytorch, please see some tutorials before working on this part.

# You should fill in the parts indicated as TODO in the 'REINFORCE' function and the methods of 'VApproximationWithNN' and 'PiApproximationWithNN' class. The 'REINFORCE' function should return the Monte-Carlo return at time 0 (i.e. ) of all the episodes that were executed during the learning. Note that these statistics can represent the progress of learning.

# To guarantee convergence, here are few specifics to follow:

# - Use AdamOptimizer with beta1=0.9, beta2=0.999 and learning rate <= 3 * 10^-4
# - Use two hidden layers with nodes each and ReLU activation function. The final layer for the policy should be a softmax over the n actions, and that for the value function should have no activation function.

# Note: In the update method of the PiApproximationWithNN class, the argument gamma_t is the value of (gamma)^t, which you will need to pass when you call this method in the REINFORCE function.



from typing import Iterable
import numpy as np
import torch

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size


        self.pi = nn.Sequential(
            nn.Linear(state_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=3e-4, betas=(0.9, 0.999))

        def __call__(self,s) -> int:
            return torch.multinomial(self.pi(torch.Tensor(s)), 1).item()

        def update(self, s, a, gamma_t, delta):
            self.optimizer.zero_grad()
            pi_s = self.pi(torch.Tensor(s))
            loss = -torch.log(pi_s[a]) * delta * gamma_t
            loss.backward()
            self.optimizer.step()



    def __call__(self,s) -> int:
        return np.random.choice(range(len(self.pi(s))), p=self.pi(s))

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.pi.update(s, a, gamma_t, delta)

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.V = nn.Sequential(
            nn.Linear(state_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.V.parameters(), lr=3e-4, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        return self.V.update(s, G)

    def update(self,s,G):
        def update(self, s, G):
            self.optimizer.zero_grad()
            V_s = self.V(torch.Tensor(s))
            loss = (G - V_s)**2
            loss.backward()
            self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    returns = []  # list to store the Monte-Carlo returns at time 0 for each episode

    for episode in range(num_episodes):
        episode_returns = 0  # initialize the return for the current episode
        state = env.reset()  # reset the environment and get the initial state

        # lists to store the states, actions, and rewards for the current episode
        states = []
        actions = []
        rewards = []

        done = False  # flag to indicate if the episode is done

        while not done:
            action = pi(state)  # select an action based on the policy
            next_state, reward, done, _ = env.step(action)  # take a step in the environment

            # store the current state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state  # update the current state

        # calculate the Monte-Carlo return at time 0 for the current episode
        G_0 = 0
        for t in range(len(states)):
            G_0 += gamma**t * rewards[t]
        returns.append(G_0)

        # update the value function using the Monte-Carlo return
        for t in range(len(states)):
            delta = G_0 - V(states[t])  # calculate the TD error
            V.update(states[t], G_0)  # update the value function

            # update the policy using the TD error and the discount factor
            gamma_t = gamma**t
            pi.update(states[t], actions[t], gamma_t, delta)

    return returns

