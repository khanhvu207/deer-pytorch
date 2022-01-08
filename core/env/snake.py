import copy
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC
from core.environment import Environment


class MyEnv(Environment, ABC):
    VALIDATION_MODE = 0

    def __init__(self, device, debug=False, **kwargs):
        self.device = device
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._size_maze = 8
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self.intern_dim = 2
        self.debug = debug

        self.default_agent_pos = [3, 3]
        self.create_map()

    def create_map(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        self._pos_agent = self.default_agent_pos
        self._pos_goal = [self._size_maze - 2, self._size_maze - 2]
        self._map[-1, :] = 1
        self._map[0, :] = 1
        self._map[:, 0] = 1
        self._map[:, -1] = 1

    def reset(self, mode):
        self.create_map()

        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1

        self._pos_agent = self.default_agent_pos
        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def act(self, action) -> float:
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier
            included between 0 included and nActions() excluded.
        """
        self._cur_action = action
        new_x, new_y = self._pos_agent[0], self._pos_agent[1]

        def _modulo(x, base):
            return (x + base) % base

        # NOTE: x is vertical axis
        if action == 0:
            # new_x = _modulo(new_x - 1, self._size_maze)
            new_x -= 1
        elif action == 1:
            # new_x = _modulo(new_x + 1, self._size_maze)
            new_x += 1
        elif action == 2:
            # new_y = _modulo(new_y - 1, self._size_maze)
            new_y -= 1
        elif action == 3:
            # new_y = _modulo(new_y + 1, self._size_maze)
            new_y += 1

        # If the next position is unoccupied
        if self._map[new_x, new_y] == 0:
            self._pos_agent[0] = new_x
            self._pos_agent[1] = new_y

        # There is no reward in this simple environment
        reward = 0
        self._mode_score += reward

        return reward

    def get_input_dims(self):
        if self._higher_dim_obs:
            return [(1, self._size_maze * 6, self._size_maze * 6)]
        else:
            return [(1, self._size_maze, self._size_maze)]

    def observation_type(self, subject):
        return np.float

    def get_num_action(self):
        return 4

    def observe(self):
        obs = copy.deepcopy(self._map)
        obs[self._pos_agent[0], self._pos_agent[1]] = 0.5

        if self._higher_dim_obs:
            obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])

        if self.debug:
            plt.imshow(obs, cmap="gray_r")
            plt.show()

        return [obs]

    def get_higher_dim_obs(self, indices_agent, indices_reward):
        """Obtain the high-dimensional observation from indices of the agent position and the indices of the reward
        positions."""
        obs = copy.deepcopy(self._map)
        obs = obs / 1.0
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)

        # Agent representation
        agent_obs = np.zeros((6, 6))
        agent_obs[0, 2] = 0.7
        agent_obs[1, 0:5] = 0.8
        agent_obs[2, 1:4] = 0.8
        agent_obs[3, 1:4] = 0.8
        agent_obs[4, 1] = 0.8
        agent_obs[4, 3] = 0.8
        agent_obs[5, 0:2] = 0.8
        agent_obs[5, 3:5] = 0.8

        # Reward representation
        reward_obs = np.zeros((6, 6))

        for i in indices_reward:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = reward_obs

        for i in indices_agent:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = agent_obs

        return obs

    def in_terminal_state(self) -> bool:
        # Uncomment the following lines to add some cases where the episode terminates.
        # This is used to show how the environment representation interpret cases where
        # part of the environment could not be explored.
        #        if((self._pos_agent[0]<=1 and self._cur_action==0) ):
        #            return True
        return False

    def summarize_performance(self, test_data_set, *args, **kwargs):
        pass

    def end(self):
        pass


if __name__ == "__main__":
    env = MyEnv(higher_dim_obs=True, debug=True, device="cuda")
    print(env.observe())
