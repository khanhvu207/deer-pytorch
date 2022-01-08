"""
CRAR Neural network using PyTorch
"""

import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        if self._high_int_dim == True:
            self.n_channels_internal_dim = kwargs["internal_dim"]  # dim[-3]
        else:
            self.internal_dim = kwargs["internal_dim"]  # 2 for laby, 3 for catcher

        self.eps = 0.000001

    def encoder_model(self):
        """Instantiate a PyTorch model for the encoder of the CRAR learning algorithm.

        The model takes the following as input
        s : list of objects
            Each object is an array that relates to one of the observations
            with size (batch_size * history size * size of punctual observation (which is 2D, 1D or scalar)).

        Parameters:

        Returns:
        nn.Module: Returning a model that ouputs the encoding of s (=x)
        """

        self._pooling_encoder = 1

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
                # self.deep_fc_encoder = nn.Sequential(
                #     nn.Linear(200, 100),
                #     self.gate,
                #     nn.Linear(100, 50),
                #     self.gate,
                #     nn.Linear(50, 10),
                #     self.gate,
                #     nn.Linear(10, internal_dim),
                # )
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

        model = Encoder(self.internal_dim, self._input_dimensions)

        return model

    def encoder_diff_model(self, encoder_model, s1, s2):
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

        return enc_s1 - enc_s2

    def transition_model(self):
        """Instantiate a Keras model for the transition between two encoded pseudo-states.

        The model takes as inputs:
        x : internal state
        a : int
            the action considered

        Parameters
        -----------

        Returns
        -------
        model that outputs the transition of (x,a)

        """

        # MLP Transition model
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

                # self.deep_fc_encoder = nn.Sequential(
                #     nn.Linear(internal_dim + n_actions, 10),
                #     self.gate,
                #     nn.Linear(10, 30),
                #     self.gate,
                #     nn.Linear(30, 30),
                #     self.gate,
                #     nn.Linear(30, 10),
                #     self.gate,
                #     nn.Linear(10, internal_dim),
                # )

                self.internal_dim = internal_dim

            def forward(self, x):
                init_state = x[:, : self.internal_dim]
                x = self.deep_fc_encoder(x)
                x = x + init_state
                return x

            def predict(self, x):
                return self.forward(x)

        model = Transition(self.internal_dim, self._n_actions)
        return model

    def diff_Tx_x_(
        self,
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
        Tx = transition_model(torch.cat((enc_s1, action), -1))
        
        return (Tx - enc_s2) * (not_terminal)

    def force_features(self, states, actions, encoder_model, transition_model):
        raise NotImplementedError

    def float_model(self):
        """Instantiate a Keras model for fitting a float from x.

        The model takes the following inputs:
        x : internal state
        a : int
            the action considered at x

        Parameters
        -----------

        Returns
        -------
        model that outputs a float

        """

        class FloatModel(nn.Module):
            def __init__(self, internal_dim, n_actions):
                super(FloatModel, self).__init__()
                self.gate = nn.Tanh()
                self.deep_fc_encoder = nn.Sequential(
                    nn.Linear(internal_dim + n_actions, 10),
                    self.gate,
                    nn.Linear(10, 50),
                    self.gate,
                    nn.Linear(50, 20),
                    self.gate,
                    nn.Linear(20, 1),
                )

            def forward(self, x):
                x = self.deep_fc_encoder(x)
                return x

            def predict(self, x):
                return self.forward(x)

        model = FloatModel(self.internal_dim, self._n_actions)

        return model

    def full_float_model(
        self, x, action, encoder_model, float_model, plan_depth=0, transition_model=None
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

    def Q_model(self):
        """Instantiate a  a Keras model for the Q-network from x.
        The model takes the following inputs:
        x : internal state
        Parameters
        -----------

        Returns
        -------
        model that outputs the Q-values for each action
        """

        class QFunction(nn.Module):
            def __init__(self, internal_dim, n_actions):
                super(QFunction, self).__init__()
                self.gate = nn.Tanh()
                self.deep_fc_encoder = nn.Sequential(
                    nn.Linear(internal_dim, 20),
                    self.gate,
                    nn.Linear(20, 50),
                    self.gate,
                    nn.Linear(50, 20),
                    self.gate,
                    nn.Linear(20, n_actions),
                )

            def forward(self, x):
                x = self.deep_fc_encoder(x)
                return x

            def predict(self, x):
                return self.forward(x)

        model = QFunction(self.internal_dim, self._n_actions)

        return model

    def full_Q_model(
        self,
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
        Q_estim = Q_model(out)
        return Q_estim


if __name__ == "__main__":
    pass
