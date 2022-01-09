import abc
import numpy as np


class Policy(abc.ABC):
    """Abstract class for all policies.
    A policy takes observations as input, and outputs an action.

    Parameters
    -----------
    learning_algo: object from class LearningALgo
    n_actions: int or list
        Definition of the action space provided by Environment.get_num_action()
    random_state : numpy random number generator
    """

    def __init__(self, learning_algo, n_actions, random_state):
        self.learning_algo = learning_algo
        self.n_actions = n_actions
        self.random_state = random_state

        pass

    def best_action(self, state, mode=None, *args, **kwargs):
        """ Returns the best Action for the given state. This is an additional encapsulation for q-network.
        """
        action, v = self.learning_algo.chooseBestAction(state, mode, *args, **kwargs)
        return action, v

    def random_action(self):
        """ Returns a random action
        """
        if isinstance(self.n_actions, int):
            # Discrete set of actions [0, n_actions]
            action = self.random_state.randint(0, self.n_actions)
        else:
            # Continuous set of actions
            action = []
            for a in self.n_actions:
                action.append(self.random_state.uniform(a[0], a[1]))
            action = np.array(action)

        v = 0
        return action, v

    @abc.abstractmethod
    def action(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        pass


class EpsilonGreedyPolicy(Policy):
    """The policy acts greedily with probability :math:`1-\epsilon` and acts randomly otherwise.
    It is now used as a default policy for the neural agent.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    """

    def __init__(self, learning_algo, n_actions, random_state, epsilon):
        Policy.__init__(self, learning_algo, n_actions, random_state)
        self._epsilon = epsilon

    def action(self, state, mode=None, *args, **kwargs):
        if self.random_state.rand() < self._epsilon:
            action, v = self.random_action()
        else:
            action, v = self.best_action(state, mode, *args, **kwargs)

        return action, v

    def set_epsilon(self, e):
        """ Set the epsilon used for :math:`\epsilon`-greedy exploration
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon
