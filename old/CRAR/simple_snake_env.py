""" Simple maze environment
"""
import math
import numpy as np
import pdb
import torch

from deer.base_classes import Environment

import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
import copy


class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, device, **kwargs):
        self.device = device
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._size_maze = 8
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self.create_map()
        self.intern_dim = 2

    def create_map(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        # self._map[-1, :] = 1
        # self._map[0, :] = 1
        # self._map[:, 0] = 1
        # self._map[:, -1] = 1
        self._pos_agent = [3, 1]
        self._pos_goal = [self._size_maze - 2, self._size_maze - 2]

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

        # Setting the starting position of the agent
        # self._pos_agent = [self._size_maze // 2, self._size_maze // 2]
        self._pos_agent = [3, 1]

        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def act(self, action):
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
            # new_x = new_x - 1
            new_x = _modulo(new_x - 1, self._size_maze)
        elif action == 1:
            new_x = _modulo(new_x + 1, self._size_maze) 
            # new_x = new_x + 1
        elif action == 2:
            new_y = _modulo(new_y - 1, self._size_maze)
            # new_y -= 1
        elif action == 3:
            # new_y += 1
            new_y = _modulo(new_y + 1, self._size_maze)

        if self._map[new_x, new_y] == 0:
            self._pos_agent[0] = new_x
            self._pos_agent[1] = new_y

        # There is no reward in this simple environment
        reward = 0

        self._mode_score += reward
        return reward

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        """Plot of the low-dimensional representation of the environment built by the model"""

        all_possib_inp = (
            []
        )  # Will store all possible inputs (=observation) for the CRAR agent
        labels_maze = []
        self.create_map()
        for y_a in range(self._size_maze):
            for x_a in range(self._size_maze):
                state = copy.deepcopy(self._map)
                # state[self._size_maze // 2, self._size_maze // 2] = 0

                if state[x_a, y_a] == 0:
                    if self._higher_dim_obs == True:
                        all_possib_inp.append(
                            self.get_higher_dim_obs([[x_a, y_a]], [self._pos_goal])
                        )
                    else:
                        state[x_a, y_a] = 0.5
                        all_possib_inp.append(state)

        all_possib_inp = np.expand_dims(np.array(all_possib_inp, dtype="float"), axis=1)

        all_possib_inp = torch.from_numpy(all_possib_inp).float().to(self.device)
        all_possib_abs_states = learning_algo.encoder.predict(all_possib_inp)

        n = 1000
        historics = []
        for i, observ in enumerate(test_data_set.observations()[0][0:n]):
            historics.append(np.expand_dims(observ, axis=0))
        historics = np.array(historics)

        historics = torch.from_numpy(historics).float().to(self.device)
        abs_states = learning_algo.encoder.predict(historics)

        actions = test_data_set.actions()[0:n]

        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print(
            "== Mean score per episode is {} over {} episodes ==".format(
                self._mode_score / (self._mode_episode_count + 0.0001),
                self._mode_episode_count,
            )
        )

        m = cm.ScalarMappable(cmap=cm.jet)

        abs_states = abs_states.detach().cpu().numpy()
        all_possib_abs_states = all_possib_abs_states.detach().cpu().numpy()

        x = np.array(abs_states)[:, 0]
        y = np.array(abs_states)[:, 1]
        if self.intern_dim > 2:
            z = np.array(abs_states)[:, 2]

        fig = plt.figure()
        if self.intern_dim == 2:
            ax = fig.add_subplot(111)
            ax.set_xlabel(r"$X_1$")
            ax.set_ylabel(r"$X_2$")

        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel(r"$X_1$")
            ax.set_ylabel(r"$X_2$")
            ax.set_zlabel(r"$X_3$")

        # Plot the estimated transitions
        for i in range(n - 1):
            # pdb.set_trace()
            predicted1 = (
                learning_algo.transition.predict(
                    torch.cat(
                        (
                            torch.from_numpy(abs_states[i : i + 1]).float(),
                            torch.from_numpy(np.array([[1, 0, 0, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted2 = (
                learning_algo.transition.predict(
                    torch.cat(
                        (
                            torch.from_numpy(abs_states[i : i + 1]).float(),
                            torch.from_numpy(np.array([[0, 1, 0, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted3 = (
                learning_algo.transition.predict(
                    torch.cat(
                        (
                            torch.from_numpy(abs_states[i : i + 1]).float(),
                            torch.from_numpy(np.array([[0, 0, 1, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted4 = (
                learning_algo.transition.predict(
                    torch.cat(
                        (
                            torch.from_numpy(abs_states[i : i + 1]).float(),
                            torch.from_numpy(np.array([[0, 0, 0, 1]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )

            def _polar2euclid(r, t):
                return r * np.cos(t), r * np.sin(t)

            if self.intern_dim == 2:
                # r1, t1 = x[i : i + 1], y[i : i + 1]
                # x1, y1 = _polar2euclid(r1, t1)
                # r2, t2 = predicted1[0, :1], predicted1[0, 1:2]
                # x2, y2 = _polar2euclid(r2, t2)
                ax.plot(
                    # np.concatenate([x1, x2]),
                    # np.concatenate([y1, y2]),
                    np.concatenate([x[i : i + 1], predicted1[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted1[0, 1:2]]),
                    color="0.9",
                    alpha=0.75,
                )
                # r2, t2 = predicted2[0, :1], predicted2[0, 1:2]
                # x2, y2 = _polar2euclid(r2, t2)
                ax.plot(
                    # np.concatenate([x1, x2]),
                    # np.concatenate([y1, y2]),
                    np.concatenate([x[i : i + 1], predicted2[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted2[0, 1:2]]),
                    color="0.65",
                    alpha=0.75,
                )
                # r2, t2 = predicted3[0, :1], predicted3[0, 1:2]
                # x2, y2 = _polar2euclid(r2, t2)
                ax.plot(
                    # np.concatenate([x1, x2]),
                    # np.concatenate([y1, y2]),
                    np.concatenate([x[i : i + 1], predicted3[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted3[0, 1:2]]),
                    color="0.4",
                    alpha=0.75,
                )
                # r2, t2 = predicted4[0, :1], predicted4[0, 1:2]
                # x2, y2 = _polar2euclid(r2, t2)
                ax.plot(
                    # np.concatenate([x1, x2]),
                    # np.concatenate([y1, y2]),
                    np.concatenate([x[i : i + 1], predicted4[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted4[0, 1:2]]),
                    color="0.15",
                    alpha=0.75,
                )
            else:
                ax.plot(
                    np.concatenate([x[i : i + 1], predicted1[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted1[0, 1:2]]),
                    np.concatenate([z[i : i + 1], predicted1[0, 2:3]]),
                    color="0.9",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i : i + 1], predicted2[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted2[0, 1:2]]),
                    np.concatenate([z[i : i + 1], predicted2[0, 2:3]]),
                    color="0.65",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i : i + 1], predicted3[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted3[0, 1:2]]),
                    np.concatenate([z[i : i + 1], predicted3[0, 2:3]]),
                    color="0.4",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i : i + 1], predicted4[0, :1]]),
                    np.concatenate([y[i : i + 1], predicted4[0, 1:2]]),
                    np.concatenate([z[i : i + 1], predicted4[0, 2:3]]),
                    color="0.15",
                    alpha=0.75,
                )

        # Plot the dots at each time step depending on the action taken 

        if self.intern_dim == 2:
            ax.scatter(
                all_possib_abs_states[:, 0],
                all_possib_abs_states[:, 1],
                marker="x",
                edgecolors="k",
                alpha=0.5,
                s=50,
            )
        else:
            ax.scatter(
                all_possib_abs_states[:, 0],
                all_possib_abs_states[:, 1],
                all_possib_abs_states[:, 2],
                marker="x",
                depthshade=True,
                edgecolors="k",
                alpha=0.5,
                s=50,
            )

        if self.intern_dim == 2:
            axes_lims = [ax.get_xlim(), ax.get_ylim()]
        else:
            axes_lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]

        # Plot the legend for transition estimates
        box1b = TextArea(
            " Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k")
        )
        box2b = DrawingArea(90, 20, 0, 0)
        el1b = Rectangle((5, 10), 15, 2, fc="0.9", alpha=0.75)
        el2b = Rectangle((25, 10), 15, 2, fc="0.65", alpha=0.75)
        el3b = Rectangle((45, 10), 15, 2, fc="0.4", alpha=0.75)
        el4b = Rectangle((65, 10), 15, 2, fc="0.15", alpha=0.75)
        box2b.add_artist(el1b)
        box2b.add_artist(el2b)
        box2b.add_artist(el3b)
        box2b.add_artist(el4b)

        boxb = HPacker(children=[box1b, box2b], align="center", pad=0, sep=5)

        anchored_box = AnchoredOffsetbox(
            loc=3,
            child=boxb,
            pad=0.0,
            frameon=True,
            bbox_to_anchor=(0.0, 0.98),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        ax.add_artist(anchored_box)

        # plt.show()
        plt.savefig("fig_base" + str(learning_algo.update_counter) + ".pdf")

        matplotlib.pyplot.close("all")  # avoids memory leaks

    def inputDimensions(self):
        if self._higher_dim_obs == True:
            return [(1, self._size_maze * 6, self._size_maze * 6)]
        else:
            return [(1, self._size_maze, self._size_maze)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return 4

    def observe(self):
        obs = copy.deepcopy(self._map)

        obs[self._pos_agent[0], self._pos_agent[1]] = 0.5
        if self._higher_dim_obs == True:
            "self._pos_agent"
            self._pos_agent
            obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])
        # plt.imshow(obs, cmap='gray_r')
        # plt.show()
        return [obs]

    def get_higher_dim_obs(self, indices_agent, indices_reward):
        """Obtain the high-dimensional observation from indices of the agent position and the indices of the reward positions."""
        obs = copy.deepcopy(self._map)
        obs = obs / 1.0
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)

        # agent repr
        agent_obs = np.zeros((6, 6))
        agent_obs[0, 2] = 0.7
        agent_obs[1, 0:5] = 0.8
        agent_obs[2, 1:4] = 0.8
        agent_obs[3, 1:4] = 0.8
        agent_obs[4, 1] = 0.8
        agent_obs[4, 3] = 0.8
        agent_obs[5, 0:2] = 0.8
        agent_obs[5, 3:5] = 0.8

        # reward repr
        reward_obs = np.zeros((6, 6))

        for i in indices_reward:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = reward_obs

        for i in indices_agent:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = agent_obs

        # plt.imshow(obs, cmap='gray_r')
        # plt.show()
        return obs

    def inTerminalState(self):
        # Uncomment the following lines to add some cases where the episode terminates.
        # This is used to show how the environment representation interpret cases where
        # part of the environment could not be explored.
        #        if((self._pos_agent[0]<=1 and self._cur_action==0) ):
        #            return True
        return False

        # If there is a goal, then terminates the environment when the goas is reached.
        # if (self._pos_agent==self._pos_goal):
        #    return True
        # else:
        #    return False


if __name__ == "__main__":
    env = MyEnv(rng=None, higher_dim_obs=False, device="cuda")
    print(env.observe())
