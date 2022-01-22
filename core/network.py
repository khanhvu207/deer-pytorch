"""
CRAR Neural network using PyTorch
"""
import math

import torch
import torch.nn as nn

from core.utils.helper_functions import polar2euclid


class Encoder(nn.Module):
    def __init__(self, internal_dim, input_dim):
        super(Encoder, self).__init__()
        self.num_channel, self.h, self.w = input_dim[0]

        self.gate = nn.Tanh()
        self.hidden = 256
        self.fc_low_dim = nn.Sequential(
            nn.Linear(self.num_channel * self.h * self.w, self.hidden),
            self.gate,
        )
        self.deep_fc_encoder = nn.Sequential(
            nn.Linear(self.hidden, 128),
            self.gate,
            nn.Linear(128, 64),
            self.gate,
            nn.Linear(64, 16),
            self.gate,
            nn.Linear(16, internal_dim),
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channel,
                out_channels=8,
                kernel_size=(2, 2),
                padding="same",
            ),
            self.gate,
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(2, 2),
                padding="same",
            ),
            self.gate,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding="same",
            ),
            self.gate,
            nn.MaxPool2d(kernel_size=3),
        )
        self.flatten_dim_after_conv = (
                32 * (self.h // 2 // 3) * (self.w // 2 // 3)
        )
        self.fc_after_conv = nn.Sequential(
            nn.Linear(self.flatten_dim_after_conv, 200), self.gate
        )

    def forward(self, x):
        if self.h <= 12 and self.w <= 12:
            x = torch.flatten(x, start_dim=1)
            x = self.fc_low_dim(x)
            x = self.deep_fc_encoder(x)
        else:
            x = self.conv_encoder(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc_after_conv(x)
            x = self.deep_fc_encoder(x)

        return x

    def predict(self, s):
        return self.forward(s)


class Transition(nn.Module):
    def __init__(self, internal_dim, n_actions):
        super(Transition, self).__init__()
        self.gate = nn.Tanh()

        self.deep_fc_encoder = nn.Sequential(
            nn.Linear(internal_dim + n_actions, 32),
            self.gate,
            nn.Linear(32, 128),
            self.gate,
            nn.Linear(128, 128),
            self.gate,
            nn.Linear(128, 32),
            self.gate,
            nn.Linear(32, internal_dim),
        )

        self.action_mask = nn.Sequential(
            nn.Linear(n_actions, 64),
            self.gate,
            nn.Linear(64, internal_dim)
        )

        self.internal_dim = internal_dim

    def forward(self, x):
        init_state = x[:, :self.internal_dim]
        dx = self.deep_fc_encoder(x)
        x = dx + init_state

        # Take the modulo of 2*pi for the second neuron
        x[:, 1] = torch.fmod(x[:, 1], 2 * math.pi)

        return x

    def predict(self, x):
        return self.forward(x)


class FloatModel(nn.Module):
    def __init__(self, internal_dim, n_actions):
        super(FloatModel, self).__init__()
        self.gate = nn.Tanh()
        self.deep_fc_encoder = nn.Sequential(
            nn.Linear(internal_dim + n_actions, 16),
            self.gate,
            nn.Linear(16, 64),
            self.gate,
            nn.Linear(64, 32),
            self.gate,
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.deep_fc_encoder(x)
        return x

    def predict(self, x):
        return self.forward(x)


class QFunction(nn.Module):
    def __init__(self, internal_dim, n_actions):
        super(QFunction, self).__init__()
        self.gate = nn.Tanh()
        self.deep_fc_encoder = nn.Sequential(
            nn.Linear(internal_dim, 32),
            self.gate,
            nn.Linear(32, 64),
            self.gate,
            nn.Linear(64, 32),
            self.gate,
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        x = self.deep_fc_encoder(x)
        return x

    def predict(self, x):
        return self.forward(x)


def encoder_diff(encoder_model, s1, s2):
    """Instantiate a Keras model that provides the difference between two encoded pseudo-states

    The model takes the two following inputs:
    s1 : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
    s2 : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).

    Parameters
    -----------
    encoder_model: instantiation of a Keras model for the encoder

    Returns
    -------
    model with output the difference between the encoding of s1 and the encoding of s2
    """

    enc_s1 = encoder_model(s1)
    enc_s2 = encoder_model(s2)

    # r1, t1 = enc_s1[:, 0], enc_s1[:, 1]
    # r2, t2 = enc_s2[:, 0], enc_s2[:, 1]
    # loss = (
    #     (r1 ** 2 + r2 ** 2 - 2.0 * r1 * r2 * torch.cos(t1 - t2))
    #     .clamp(self.eps, 100.0)
    #     .sqrt()
    # )
    s1_x, s1_y = enc_s1[:, 0] * torch.cos(enc_s1[:, 1]), enc_s1[:, 0] * torch.sin(enc_s1[:, 1])
    s2_x, s2_y = enc_s2[:, 0] * torch.cos(enc_s2[:, 1]), enc_s2[:, 0] * torch.sin(enc_s2[:, 1])
    loss = torch.zeros_like(enc_s1)
    loss[:, 0] = s1_x - s2_x
    loss[:, 1] = s1_y - s2_y
    return loss
    # return enc_s1 - enc_s2


def diff_tx_x(
        s1,
        s2,
        action,
        not_terminal,
        encoder_model,
        transition_model,
        plan_depth=0,
):
    """For plan_depth=0, instantiate a Keras model that provides the difference between T(E(s1),a) and E(s2).
    Note that it gives 0 if the transition leading to s2 is terminal (we don't need to fit the transition if
    it is terminal).

    For plan_depth=0, the model takes the four following inputs:
    s1 : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
    s2 : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
    a : list of ints with length (plan_depth+1)
        the action(s) considered at s1
    terminal : boolean
        Whether the transition leading to s2 is terminal

    Parameters
    -----------
    encoder_model: instantiation of a Keras model for the encoder (E)
    transition_model: instantiation of a Keras model for the transition (T)
    plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2
    (input a is then a list of actions)

    Returns
    -------
    model with output Tx (= model estimate of x')

    """
    enc_s1 = encoder_model(s1)
    enc_s2 = encoder_model(s2)
    tx = transition_model(torch.cat((enc_s1, action), -1))

    # return (tx - enc_s2) * not_terminal

    # r1, t1 = tx[:, 0], tx[:, 1]
    # r2, t2 = enc_s2[:, 0], enc_s2[:, 1]
    # loss = (
    #     (r1 ** 2 + r2 ** 2 - 2.0 * r1 * r2 * torch.cos(t1 - t2))
    #     .clamp(self.eps, 100.0)
    #     .sqrt()
    # )

    # translate to euclidean coordinate and compute the L2 loss
    t_x, t_y = tx[:, 0] * torch.cos(tx[:, 1]), tx[:, 0] * torch.sin(tx[:, 1])
    s2_x, s2_y = enc_s2[:, 0] * torch.cos(enc_s2[:, 1]), enc_s2[:, 0] * torch.sin(enc_s2[:, 1])
    loss = torch.zeros_like(tx)
    loss[:, 0] = t_x - s2_x
    loss[:, 1] = t_y - s2_y

    return loss * not_terminal


def full_float_model(
        x, action, encoder_model, float_model, plan_depth=0, transition_model=None
):
    """Instantiate a Keras model for fitting a float from s.

    The model takes the four following inputs:
    s : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
    a : list of ints with length (plan_depth+1)
        the action(s) considered at s

    Parameters
    -----------
    encoder_model: instantiation of a Keras model for the encoder (E)
    float_model: instantiation of a Keras model for fitting a float from x
    plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s
    (input a is then a list of actions)
    transition_model: instantiation of a Keras model for the transition (T)

    Returns
    -------
    model with output the reward r
    """

    enc_x = encoder_model(x)
    reward_pred = float_model(torch.cat((enc_x, action), -1))
    return reward_pred


def full_q_model(
        x,
        encoder_model,
        Q_model,
        plan_depth=0,
        transition_model=None,
        R_model=None,
        discount_model=None,
):
    """Instantiate a  a Keras model for the Q-network from s.
    The model takes the following inputs:
    s : list of objects
        Each object is a numpy array that relates to one of the observations
        with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
    a : list of ints with length plan_depth; if plan_depth=0, there isn't any input for a.
        the action(s) considered at s

    Parameters
    -----------
    encoder_model: instantiation of a Keras model for the encoder (E)
    Q_model: instantiation of a Keras model for the Q-network from x.
    plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s
    (input a is then a list of actions)
    transition_model: instantiation of a Keras model for the transition (T)
    R_model: instantiation of a Keras model for the reward
    discount_model: instantiation of a Keras model for the discount

    Returns
    -------
    model with output the Q-values
    """

    out = encoder_model(x)
    q_estimation = Q_model(out)
    return q_estimation


class NN:
    """
    Deep Q-learning network using Pytorch

    Parameters
    -----------
    batch_size (int) : Number of tuples taken into account for each iteration of gradient descent
    input_dimensions (Tuple) : A tuple denotes the input size
    n_actions (int) : Number of possible actions
    random_state : Numpy random number generator
    high_int_dim (Boolean) : Whether the abstract state should be high dimensional in the form of frames/vectors or whether it should be low-dimensional
    """

    def __init__(self, batch_size, input_dimensions, n_actions, random_state, **kwargs):
        self._input_dimensions = input_dimensions
        self._batch_size = batch_size
        self._random_state = random_state
        self._n_actions = n_actions
        self._high_int_dim = kwargs["high_int_dim"]

        if self._high_int_dim:
            self.n_channels_internal_dim = kwargs["internal_dim"]
        else:
            self.internal_dim = kwargs["internal_dim"]

        self.eps = 0.000001

        # Networks
        self.encoder = Encoder(internal_dim=self.internal_dim, input_dim=self._input_dimensions)
        self.transition = Transition(internal_dim=self.internal_dim, n_actions=self._n_actions)
        self.float_model = FloatModel(internal_dim=self.internal_dim, n_actions=self._n_actions)
        self.q_function = QFunction(internal_dim=self.internal_dim, n_actions=self._n_actions)

    def force_features(self, states, actions, encoder_model, transition_model):
        raise NotImplementedError


if __name__ == "__main__":
    pass
