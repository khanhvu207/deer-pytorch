"""
This module defines the base class for the learning algorithms.

"""
import pickle
import datetime
import copy
import pprint

import torch.optim
from rich.console import Console

from core.utils.seed_everything import *

from torch import optim

from core.network import NN
from core.agent import AgentError
from core.environment import Environment
from core.utils.helper_functions import (
    mean_squared_error_p,
    exp_dec_error,
)

from core.network import (
    encoder_diff,
    encoder_diff_angular,
    diff_tx_x,
    full_float_model,
    full_q_model,
)

console = Console()


class LearningAlgo(object):
    """All the Q-networks, actor-critic networks, etc. should inherit this interface.

    Parameters
    -----------
    environment : object from class Environment
        The environment linked to the Q-network
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    """

    def __init__(self, environment: Environment, batch_size):
        self._environment = environment
        self._df = 0.9
        self._lr = 0.005
        self._input_dimensions = self._environment.get_input_dims()
        self._n_actions = self._environment.get_num_action()
        self._batch_size = batch_size

    def train(self, states, actions, rewards, nextStates, terminals):
        """This method performs the training step (e.g. using Bellman iteration in a deep Q-network)
        for one batch of tuples.
        """
        raise NotImplementedError()

    def chooseBestAction(self, state):
        """Get the best action for a pseudo-state"""
        raise NotImplementedError()

    def qValues(self, state):
        """Get the q value for one pseudo-state"""
        raise NotImplementedError()

    def setLearningRate(self, lr):
        """Setting the learning rate
        NB: The learning rate has usually to be set in the optimizer, hence this function should
        be overridden. Otherwise, the learning rate change is likely not to be taken into account

        Parameters
        -----------
        lr : float
            The learning rate that has to bet set
        """
        self._lr = lr

    def setDiscountFactor(self, df):
        """Setting the discount factor

        Parameters
        -----------
        df : float
            The discount factor that has to bet set
        """
        if df < 0.0 or df > 1.0:
            raise AgentError("The discount factor should be in [0,1]")

        self._df = df

    def learningRate(self):
        """Getting the learning rate"""
        return self._lr

    def discountFactor(self):
        """Getting the discount factor"""
        return self._df


class CRAR(LearningAlgo):
    """
    Combined Reinforcement learning via Abstract Representations (CRAR) using Keras

    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        Default is deer.learning_algos.NN_keras
    """

    def __init__(
            self,
            environment,
            rho=0.9,
            rms_epsilon=0.0001,
            momentum=0,
            clip_norm=1.0,
            C=5,
            radius=1,
            beta1=1.0,
            beta2=0.0,
            freeze_interval=1000,
            batch_size=32,
            update_rule="rmsprop",
            seed=2022,
            num_training_steps=None,
            random_state=np.random.RandomState(),
            double_Q=False,
            neural_network=NN,
            wandb_logger=None,
            device="cpu",
            print_every=100,
            **kwargs,
    ):
        """Initialize the environment"""
        LearningAlgo.__init__(self, environment, batch_size)
        self.device = device
        self.print_every = print_every

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._C = C
        self._radius = radius
        self._random_state = random_state
        self.update_counter = 0
        self._high_int_dim = kwargs.get("high_int_dim", False)
        self._internal_dim = kwargs.get("internal_dim", 2)
        self._beta1 = beta1
        self._beta2 = beta2
        self.seed = seed
        self.num_training_steps = num_training_steps
        self.wandb_logger = wandb_logger
        self.loss_interpret = 0
        self.loss_alignment = 0
        self.lossR = 0
        self.loss_q = 0
        self.loss_csc = 0
        self.loss_hypercube = 0
        self.loss_uniformity = 0
        self.loss_force_feature = 0
        self.loss_gamma = 0
        self.loss_reconstruct = 0

        self.learn_and_plan = neural_network(
            self._batch_size,
            self._input_dimensions,
            self._n_actions,
            self._random_state,
            high_int_dim=self._high_int_dim,
            internal_dim=self._internal_dim,
        )
        self.encoder = self.learn_and_plan.encoder.to(self.device)
        self.decoder = self.learn_and_plan.decoder.to(self.device)
        self.R = self.learn_and_plan.float_model.to(self.device)
        self.Q = self.learn_and_plan.q_function.to(self.device)
        self.gamma = self.learn_and_plan.float_model.to(self.device)
        self.transition = self.learn_and_plan.transition.to(self.device)

        # Define optimizers
        optimizer_map = {
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW
        }
        chosen_optimizer = optimizer_map[update_rule]
        self.encoder_optimizer = chosen_optimizer(self.encoder.parameters(), lr=self._lr)
        self.decoder_optimizer = chosen_optimizer(self.decoder.parameters(), lr=self._lr)
        self.R_optimizer = chosen_optimizer(self.R.parameters(), lr=self._lr)
        self.Q_optimizer = chosen_optimizer(self.Q.parameters(), lr=self._lr)
        self.gamma_optimizer = chosen_optimizer(self.gamma.parameters(), lr=self._lr)
        self.transition_optimizer = chosen_optimizer(self.transition.parameters(), lr=self._lr)

        self.optimizers = [
            self.encoder_optimizer,
            self.decoder_optimizer,
            self.R_optimizer,
            self.Q_optimizer,
            self.gamma_optimizer,
            self.transition_optimizer
        ]

        # Watch models gradient
        wandb_logger.watch(self.encoder, log="all")
        wandb_logger.watch(self.R, log="all")
        wandb_logger.watch(self.Q, log="all")
        wandb_logger.watch(self.gamma, log="all")
        wandb_logger.watch(self.transition, log="all")

        self.all_losses_list = []
        self.training_steps = []

        self.encoder_diff = encoder_diff
        self.full_Q = full_q_model

        # used to fit rewards
        self.full_R = full_float_model

        # used to fit gamma
        self.full_gamma = full_float_model

        # used to fit transitions
        self.diff_Tx_x_ = diff_tx_x

        # constraint on consecutive t
        self.diff_s_s_ = encoder_diff

        # used to force features variations
        if not self._high_int_dim:
            self.force_features = self.learn_and_plan.force_features

        self.all_models = [self.encoder, self.R, self.Q, self.gamma, self.transition]

        # Compile all models
        # self._compile()

        # Instantiate the same neural network as a target network.
        self.learn_and_plan_target = neural_network(
            self._batch_size,
            self._input_dimensions,
            self._n_actions,
            self._random_state,
            high_int_dim=self._high_int_dim,
            internal_dim=self._internal_dim,
        )
        self.encoder_target = self.learn_and_plan_target.encoder.to(self.device)
        self.Q_target = self.learn_and_plan_target.q_function.to(self.device)
        self.R_target = self.learn_and_plan_target.float_model.to(self.device)
        self.gamma_target = self.learn_and_plan_target.float_model.to(self.device)
        self.transition_target = self.learn_and_plan_target.transition.to(self.device)

        self.full_Q_target = full_q_model

        self.all_models_target = [
            self.encoder_target,
            self.R_target,
            self.Q_target,
            self.gamma_target,
            self.transition_target,
        ]

        self._resetQHat()

    def train(
            self, states_val, actions_val, rewards_val, next_states_val, terminals_val
    ):
        """
        Train CRAR from one batch of data.
        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).

        actions_val : numpy array of integers with size [self._batch_size]
            actions[i] is the action taken after having observed states[:][i].

        rewards_val : numpy array of floats with size [self._batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].

        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).

        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Returns
        -------
        Average loss of the batch training for the Q-values (RMSE)
        Individual (square) losses for the Q-values for each tuple
        """
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        states_val = list(states_val)
        next_states_val = list(next_states_val)
        states_val = torch.from_numpy(states_val[0]).float().to(self.device)
        next_states_val = torch.from_numpy(next_states_val[0]).float().to(self.device)
        onehot_actions = torch.from_numpy(onehot_actions).float().to(self.device)
        terminals_val = (
            torch.from_numpy(terminals_val[:, None].astype(np.int32))
                .float()
                .to(self.device)
        )
        rewards_val = (
            torch.from_numpy(rewards_val[:, None].astype(np.int32))
                .float()
                .to(self.device)
        )

        out = self.diff_Tx_x_(
            states_val,
            next_states_val,
            onehot_actions,
            (1 - terminals_val),
            self.encoder,
            self.transition,
        )
        alignment_loss = torch.nn.MSELoss()(out, torch.zeros_like(out))
        self.loss_alignment += alignment_loss.item()

        # L_infinity ball of radius 1 loss
        out = self.encoder(states_val)
        # euclid_coords = torch.zeros_like(out)
        # euclid_coords[:, 0] = out[:, 0] * torch.cos(out[:, 1])
        # euclid_coords[:, 1] = out[:, 0] * torch.sin(out[:, 1])
        hypercube_loss = mean_squared_error_p(out, radius=self._radius)
        self.loss_hypercube += hypercube_loss.item()

        # This one is very important
        # Entropy maximization loss (through exponential) between two random states
        def roll(x, n):
            return torch.cat((x[-n:], x[:-n]))

        rolled = roll(states_val, -31)
        out = self.encoder_diff(self.encoder, states_val, rolled)
        uniformity_loss = self._beta1 * exp_dec_error(out, C=self._C)
        self.loss_uniformity += uniformity_loss.item()

        # Enforce consecutive states to be w=0.1 distance apart
        out = self.encoder_diff(self.encoder, states_val, next_states_val)
        out = out.norm(dim=1)
        csc_loss = torch.maximum(out - 0.5, torch.zeros_like(out)).sum()
        self.loss_csc += csc_loss.item()

        # Calculate the loss for reward
        # self.optimizer_full_R.zero_grad()
        # out = self.full_R(states_val, onehot_actions, self.encoder, self.R)
        # loss = torch.nn.MSELoss()
        # loss_val = loss(out, rewards_val)
        # self.lossR += loss_val.data.numpy()
        # loss_val.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.encoder.parameters(), max_norm=self._clip_norm
        # )
        # torch.nn.utils.clip_grad_norm_(
        #     self.R.parameters(), max_norm=self._clip_norm
        # )
        # self.optimizer_full_R.step()

        # Calculate loss for gamma
        # self.optimizer_full_gamma.zero_grad()
        # out = self.full_gamma(states_val, onehot_actions, self.encoder, self.gamma)
        # loss = torch.nn.MSELoss()
        # loss_val = loss(out, (1 - terminals_val[:]) * self._df)
        # self.loss_gamma += loss_val.data.numpy()
        # loss_val.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.encoder.parameters(), max_norm=self._clip_norm
        # )
        # torch.nn.utils.clip_grad_norm_(
        #     self.gamma.parameters(), max_norm=self._clip_norm
        # )
        # self.optimizer_full_gamma.step()

        # Q Learning loss
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()

        next_q_vals = self.full_Q_target(
            next_states_val, self.encoder_target, self.Q_target
        ).detach()

        max_next_q_vals = torch.max(next_q_vals, dim=1)[0]
        not_terminals = 1 - terminals_val
        target = rewards_val + not_terminals * self._df * max_next_q_vals[:, None]
        q_vals = self.full_Q(states_val, self.encoder, self.Q).gather(
            1, torch.from_numpy(actions_val.astype(int)[:, None]).to(self.device)
        )
        q_loss = torch.nn.MSELoss()(q_vals, target)
        loss = q_loss.item()
        self.loss_q += loss

        # Backprop
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        total_loss = 2 * alignment_loss + hypercube_loss + uniformity_loss
        total_loss.backward()

        # (Optional) Gradient clipping
        if self._clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self._clip_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self._clip_norm)
            torch.nn.utils.clip_grad_norm_(self.transition.parameters(), max_norm=self._clip_norm)
            torch.nn.utils.clip_grad_norm_(self.R.parameters(), max_norm=self._clip_norm)
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=self._clip_norm)
            torch.nn.utils.clip_grad_norm_(self.gamma.parameters(), max_norm=self._clip_norm)

        for optimizer in self.optimizers:
            optimizer.step()

        if self.update_counter % self.print_every == 0:
            console.print(
                "Number of training steps: " + str(self.update_counter) + ".",
                style="bold red",
            )
            self.wandb_logger.log({"nr. of training steps": self.update_counter})

            all_losses = {
                "loss_alignment": self.loss_alignment,
                "loss_r": self.lossR,
                "loss_gamma": self.loss_gamma,
                "loss_q": self.loss_q,
                "loss_csc": self.loss_csc,
                "loss_hypercube": self.loss_hypercube,
                "loss_uniformity": self.loss_uniformity
            }

            pprint.pprint(all_losses)
            self.wandb_logger.log(all_losses)

            # Record the losses and training step
            self.all_losses_list.append(all_losses)
            self.training_steps.append(self.update_counter)

            self.lossR = 0
            self.loss_gamma = 0
            self.loss_q = 0
            self.loss_alignment = 0
            self.loss_interpret = 0
            self.loss_csc = 0
            self.loss_hypercube = 0
            self.loss_uniformity = 0
            self.loss_force_feature = 0
            self.loss_reconstruct = 0
            console.print()

            # Dump training logs to disk when train ends
            if self.update_counter + self.print_every == self.num_training_steps:
                log = (self.all_losses_list, self.training_steps)
                cur_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
                filename = f"log-{self.seed}-" + cur_time_str + ".pkl"
                with open(filename, "wb") as f:
                    pickle.dump(obj=log, file=f)
                    print("Logs saved!")

        self.update_counter += 1

        return np.sqrt(loss), (q_vals.detach() - target) ** 2

    def _compile(self):
        """Compile all the optimizers for the different losses"""

        if self._update_rule == "rmsprop":
            self.optimizer_full_Q = optim.RMSprop(
                list(self.encoder.parameters()) + list(self.Q.parameters()),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )
            self.optimizer_diff_Tx_x_ = optim.RMSprop(
                list(self.encoder.parameters()) + list(self.transition.parameters()),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )  # Different optimizers for each network;
            self.optimizer_full_R = optim.RMSprop(
                list(self.encoder.parameters()) + list(self.R.parameters()),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )  # to possibly modify them separately
            self.optimizer_full_gamma = optim.RMSprop(
                list(self.encoder.parameters()) + list(self.gamma.parameters()),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )
            self.optimizer_encoder = optim.RMSprop(
                self.encoder.parameters(),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )
            self.optimizer_encoder_diff = optim.RMSprop(
                self.encoder.parameters(),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )
            self.optimizer_diff_s_s_ = optim.RMSprop(
                self.encoder.parameters(),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )
            self.optimizer_force_features = optim.RMSprop(
                self.transition.parameters(),
                lr=self._lr,
                alpha=self._rho,
                eps=self._rms_epsilon,
            )

        elif self._update_rule == "adam":
            self.optimizer_full_Q = optim.Adam(
                list(self.encoder.parameters()) + list(self.Q.parameters()),
                lr=self._lr,
            )
            self.optimizer_diff_Tx_x_ = optim.Adam(
                list(self.encoder.parameters()) + list(self.transition.parameters()),
                lr=self._lr,
            )
            self.optimizer_full_R = optim.Adam(
                list(self.encoder.parameters()) + list(self.R.parameters()),
                lr=self._lr,
            )
            self.optimizer_full_gamma = optim.Adam(
                list(self.encoder.parameters()) + list(self.gamma.parameters()),
                lr=self._lr,
            )
            self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=self._lr)
            self.optimizer_encoder_diff = optim.Adam(
                self.encoder.parameters(), lr=self._lr
            )
            self.optimizer_diff_s_s_ = optim.Adam(
                self.encoder.parameters(), lr=self._lr
            )
            self.optimizer_force_features = optim.Adam(
                self.transition.parameters(), lr=self._lr
            )

        else:
            raise Exception(
                "The update_rule " + self._update_rule + " is not implemented."
            )

        self.optimizers = [
            self.optimizer_full_Q,
            self.optimizer_diff_Tx_x_,
            self.optimizer_full_R,
            self.optimizer_full_gamma,
            self.optimizer_encoder,
            self.optimizer_encoder_diff,
            self.optimizer_diff_s_s_,
        ]

    def qValues(self, state_val):
        """Get the q values for one pseudo-state (without planning)
        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).
        Returns
        -------
        The q values for the provided pseudo state
        """
        copy_state = copy.deepcopy(state_val)
        copy_state = torch.tensor(copy_state).float().to(self.device)
        return (
            self.full_Q(copy_state, encoder_model=self.encoder, Q_model=self.Q)
                .squeeze(0)
                .detach()
                .numpy()
        )

    def q_values_planning(self, state_val, depth):
        """Get the average Q-values up to planning depth d for one pseudo-state.

        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).
        depth : int
            planning depth
        Returns
        -------
        The average q values with planning depth up to d for the provided pseudo-state
        """
        q_values = []
        state_val = torch.tensor(state_val).float().to(self.device)
        abstract_state = self.encoder(state_val).detach()

        for action in range(self._n_actions):
            q_values.append(self.tree_search(abstract_state, action, depth, b=2))

        return q_values

    def tree_search(self, abstract_state, action, depth, b):
        q_value = self.Q(abstract_state).squeeze(0).detach().numpy()

        if depth == 0:
            return q_value[action]

        onehot_actions = np.identity(self._n_actions)
        onehot_cur_action = (
            torch.from_numpy(onehot_actions[action])
                .float()
                .unsqueeze(0)
                .to(self.device)
        )

        top_b_next_actions = np.argpartition(q_value, -b)[-b:]

        next_abstract_state = self.transition(
            torch.cat(
                (
                    abstract_state,
                    onehot_cur_action,
                ),
                dim=1,
            )
        ).detach()

        predicted_reward = self.R(
            torch.cat(
                (
                    abstract_state,
                    onehot_cur_action,
                ),
                dim=1,
            )
        ).item()

        predicted_gamma = self.gamma(
            torch.cat(
                (
                    abstract_state,
                    onehot_cur_action,
                ),
                dim=1,
            )
        ).item()

        next_q_values = []
        for next_action in top_b_next_actions:
            next_q_values.append(
                self.tree_search(next_abstract_state, next_action, depth=depth - 1, b=2)
            )

        return predicted_reward + predicted_gamma * np.max(next_q_values)

    def chooseBestAction(self, state, mode=0, *args, **kwargs):
        """Get the best action for a pseudo-state
        Arguments
        ---------
        state : list of numpy arrays
             One pseudo-state. The number of arrays and their dimensions matches self.environment.inputDimensions().
        mode : int
            Identifier of the mode (-1 is reserved for the training mode).
        Returns
        -------
        The best action : int
        """
        copy_state = copy.deepcopy(state)

        q_vals = self.q_values_planning(
            copy_state,
            depth=0,
        )

        return np.argmax(q_vals), np.max(q_vals)

    def _resetQHat(self):
        """Set the target Q-network weights equal to the main Q-network weights"""
        for mod, mod_t in zip(self.all_models, self.all_models_target):
            mod_t.load_state_dict(mod.state_dict())
            mod_t.eval()

    def setLearningRate(self, lr):
        """Setting the learning rate
        Parameters
        -----------
        lr : float
            The learning rate that has to be set
        """

        self._lr = lr
        print("New learning rate set to " + str(self._lr) + ".")
        for i, optim in enumerate(self.optimizers):
            for param_group in optim.param_groups:
                param_group["lr"] = lr if i != len(self.optimizers) - 1 else lr / 5.0
