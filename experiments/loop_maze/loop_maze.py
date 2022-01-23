import copy
from abc import ABC

from core.utils.seed_everything import *

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# import numpy as np
# import torch
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from matplotlib.patches import Rectangle

from core.environment import Environment
from core.utils.helper_functions import polar2euclid


class MyEnv(Environment, ABC):
    VALIDATION_MODE = 0

    def __init__(self, device, debug=False, **kwargs):
        self.device = device
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._size_maze_x = 5
        self._size_maze_y = 5
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self.intern_dim = 2
        self.debug = debug

        self.default_agent_pos = [1, 1]
        self.create_map()

    def create_map(self):
        self._map = np.zeros((self._size_maze_x, self._size_maze_y))
        self._pos_agent = self.default_agent_pos
        self._pos_goal = [self._size_maze_x - 2, self._size_maze_y - 2]
        self._map[-1, :] = 1
        self._map[0, :] = 1
        # self._map[:, 0] = 1
        # self._map[:, -1] = 1

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
        return [1 * [self._size_maze_x * [self._size_maze_y * [0]]]]

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
            new_y = _modulo(new_y - 1, self._size_maze_y)
            # new_y -= 1
        elif action == 3:
            new_y = _modulo(new_y + 1, self._size_maze_y)
            # new_y += 1

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
            return [(1, self._size_maze_x * 6, self._size_maze_y * 6)]
        else:
            return [(1, self._size_maze_x, self._size_maze_y)]

    def observation_type(self, subject):
        return np.float

    def get_num_action(self):
        return 4

    def observe(self):
        obs = copy.deepcopy(self._map)
        obs[self._pos_agent[0], self._pos_agent[1]] = 0.5

        if self._higher_dim_obs:
            obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])

        if self.debug is True:
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
            obs[i[0] * 6: (i[0] + 1) * 6:, i[1] * 6: (i[1] + 1) * 6] = reward_obs

        for i in indices_agent:
            obs[i[0] * 6: (i[0] + 1) * 6:, i[1] * 6: (i[1] + 1) * 6] = agent_obs

        return obs

    def in_terminal_state(self) -> bool:
        # Uncomment the following lines to add some cases where the episode terminates.
        # This is used to show how the environment representation interpret cases where
        # part of the environment could not be explored.
        #        if((self._pos_agent[0]<=1 and self._cur_action==0) ):
        #            return True
        return False

    def summarize_performance(self, test_data_set, learning_algo, *args, **kwargs):
        """Plot of the low-dimensional representation of the environment built by the model"""
        all_possible_inputs = []
        self.create_map()

        for y_a in range(self._size_maze_y):
            for x_a in range(self._size_maze_x):
                state = copy.deepcopy(self._map)

                if state[x_a, y_a] == 0:
                    if self._higher_dim_obs:
                        all_possible_inputs.append(
                            self.get_higher_dim_obs([[x_a, y_a]], [self._pos_goal])
                        )
                    else:
                        state[x_a, y_a] = 0.5
                        all_possible_inputs.append(state)

        all_possible_inputs = np.expand_dims(np.array(all_possible_inputs, dtype="float"), axis=1)
        all_possible_inputs = torch.from_numpy(all_possible_inputs).float().to(self.device)
        all_possible_abs_states = learning_algo.encoder(all_possible_inputs)
        all_possible_abs_states = all_possible_abs_states.detach().cpu().numpy()

        # if not self.in_terminal_state():
        #     self._mode_episode_count += 1
        # print(
        #     "== Mean score per episode is {} over {} episodes ==".format(
        #         self._mode_score / (self._mode_episode_count + 0.0001),
        #         self._mode_episode_count,
        #     )
        # )

        # cm.ScalarMappable(cmap=cm.jet)
        # abs_states = abs_states.detach().cpu().numpy()

        x = np.array(all_possible_abs_states)[:, 0]
        y = np.array(all_possible_abs_states)[:, 1]
        if self.intern_dim > 2:
            z = np.array(all_possible_abs_states)[:, 2]

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
        for i in range(len(all_possible_abs_states)):
            predicted_action0 = (
                learning_algo.transition(
                    torch.cat(
                        (
                            torch.from_numpy(all_possible_abs_states[i: i + 1]).float(),
                            torch.from_numpy(np.array([[1, 0, 0, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted_action1 = (
                learning_algo.transition(
                    torch.cat(
                        (
                            torch.from_numpy(all_possible_abs_states[i: i + 1]).float(),
                            torch.from_numpy(np.array([[0, 1, 0, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted_action2 = (
                learning_algo.transition(
                    torch.cat(
                        (
                            torch.from_numpy(all_possible_abs_states[i: i + 1]).float(),
                            torch.from_numpy(np.array([[0, 0, 1, 0]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted_action3 = (
                learning_algo.transition(
                    torch.cat(
                        (
                            torch.from_numpy(all_possible_abs_states[i: i + 1]).float(),
                            torch.from_numpy(np.array([[0, 0, 0, 1]])).float(),
                        ),
                        -1,
                    ).to(self.device)
                )
                .detach()
                .cpu()
                .numpy()
            )

            if self.intern_dim == 2:
                r1, t1 = x[i: i + 1], y[i: i + 1]
                x1, y1 = polar2euclid(r1, t1)
                r2, t2 = predicted_action0[0, :1], predicted_action0[0, 1:2]
                x2, y2 = polar2euclid(r2, t2)
                ax.plot(
                    np.concatenate([x1, x2]),
                    np.concatenate([y1, y2]),
                    # np.concatenate([x[i: i + 1], predicted1[0, :1]]),
                    # np.concatenate([y[i: i + 1], predicted1[0, 1:2]]),
                    color="0.9",
                    alpha=0.75,
                )
                r2, t2 = predicted_action1[0, :1], predicted_action1[0, 1:2]
                x2, y2 = polar2euclid(r2, t2)
                ax.plot(
                    np.concatenate([x1, x2]),
                    np.concatenate([y1, y2]),
                    # np.concatenate([x[i: i + 1], predicted2[0, :1]]),
                    # np.concatenate([y[i: i + 1], predicted2[0, 1:2]]),
                    color="0.65",
                    alpha=0.75,
                )
                r2, t2 = predicted_action2[0, :1], predicted_action2[0, 1:2]
                x2, y2 = polar2euclid(r2, t2)
                ax.plot(
                    np.concatenate([x1, x2]),
                    np.concatenate([y1, y2]),
                    # np.concatenate([x[i: i + 1], predicted3[0, :1]]),
                    # np.concatenate([y[i: i + 1], predicted3[0, 1:2]]),
                    color="0.4",
                    alpha=0.75,
                )
                r2, t2 = predicted_action3[0, :1], predicted_action3[0, 1:2]
                x2, y2 = polar2euclid(r2, t2)
                ax.plot(
                    np.concatenate([x1, x2]),
                    np.concatenate([y1, y2]),
                    # np.concatenate([x[i: i + 1], predicted4[0, :1]]),
                    # np.concatenate([y[i: i + 1], predicted4[0, 1:2]]),
                    color="0.15",
                    alpha=0.75,
                )
            else:
                ax.plot(
                    np.concatenate([x[i: i + 1], predicted_action0[0, :1]]),
                    np.concatenate([y[i: i + 1], predicted_action0[0, 1:2]]),
                    np.concatenate([z[i: i + 1], predicted_action0[0, 2:3]]),
                    color="0.9",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i: i + 1], predicted_action1[0, :1]]),
                    np.concatenate([y[i: i + 1], predicted_action1[0, 1:2]]),
                    np.concatenate([z[i: i + 1], predicted_action1[0, 2:3]]),
                    color="0.65",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i: i + 1], predicted_action2[0, :1]]),
                    np.concatenate([y[i: i + 1], predicted_action2[0, 1:2]]),
                    np.concatenate([z[i: i + 1], predicted_action2[0, 2:3]]),
                    color="0.4",
                    alpha=0.75,
                )
                ax.plot(
                    np.concatenate([x[i: i + 1], predicted_action3[0, :1]]),
                    np.concatenate([y[i: i + 1], predicted_action3[0, 1:2]]),
                    np.concatenate([z[i: i + 1], predicted_action3[0, 2:3]]),
                    color="0.15",
                    alpha=0.75,
                )

        # Plot the dots at each time step depending on the action taken
        xs, ys = polar2euclid(all_possible_abs_states[:, 0], all_possible_abs_states[:, 1])

        if self.intern_dim == 2:
            ax.scatter(
                xs,
                ys,
                # all_possible_abs_states[:, 0],
                # all_possible_abs_states[:, 1],
                marker="x",
                edgecolors="k",
                alpha=0.5,
                s=50,
            )
        else:
            ax.scatter(
                all_possible_abs_states[:, 0],
                all_possible_abs_states[:, 1],
                all_possible_abs_states[:, 2],
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

    def end(self):
        pass


if __name__ == "__main__":
    env = MyEnv(higher_dim_obs=False, debug=True, device="cuda")
    env.act(action=1)
    print(env.observe())

    # import math
    # mod_pi = lambda v: (v + math.pi) % (2 * math.pi) - math.pi
    # x = np.arange(-20, 20, step=0.1)
    # y = np.array(list(map(mod_pi, x)))
    # z = np.fmod(x, 2 * math.pi)
    # plt.plot(x, y, label="correct")
    # plt.plot(x, z, label="wrong")
    # plt.legend()
    # plt.savefig("mod_pi.png", dpi=300)

