import os
import yaml
import json
import wandb
import jsons
import pprint
import logging
# import numpy as np
from joblib import hash, dump, load


import core.controller as bc
from core.agent import NeuralAgent
from core.policy import EpsilonGreedyPolicy
from core.learning_algorithm import CRAR
from core.utils.seed_everything import *
from loop_maze import MyEnv

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load config
    with open("config.yaml") as f:
        args = yaml.safe_load(f)

    pprint.PrettyPrinter(indent=2).pprint(args)

    experiment_args = args["experiment_args"]
    env_args = args["env_args"]
    train_args = args["train_args"]
    logger_args = args["logger_args"]

    rng = (
        np.random.RandomState(2022)
        if train_args["deterministic"]
        else np.random.RandomState()
    )

    # --- Instantiate logger ---
    logger = wandb.init(
        mode="online" if logger_args["online_mode"] else "offline",
        project="deer",
        entity="kvu207",
        config=json.loads(jsons.dumps(args)),
    )

    # --- Instantiate environment ---
    env = MyEnv(
        size_x=env_args["size_x"],
        size_y=env_args["size_y"],
        device=train_args["device"],
        debug=False,
        higher_dim_obs=env_args["higher_dim_obs"],
    )

    # --- Instantiate learning_algo ---
    learning_algo = CRAR(
        environment=env,
        rho=train_args["rms_decay"],
        rms_epsilon=train_args["rms_epsilon"],
        momentum=train_args["momentum"],
        clip_norm=train_args["clip_norm"],
        beta2=train_args["beta2"],
        C=train_args["C"],
        radius=train_args["radius"],
        freeze_interval=train_args["freeze_interval"],
        batch_size=train_args["batch_size"],
        update_rule=train_args["update_rule"],
        random_state=rng,
        high_int_dim=False,
        internal_dim=train_args["internal_dim"],
        wandb_logger=logger,
        device=train_args["device"],
    )

    # --- Instantiate agent ---
    agent = NeuralAgent(
        environment=env,
        learning_algo=learning_algo,
        replay_memory_size=train_args["replay_memory_size"],
        replay_start_size=max(
            env.get_input_dims()[i][0] for i in range(len(env.get_input_dims()))
        ),
        batch_size=train_args["batch_size"],
        random_state=rng,
        train_policy=EpsilonGreedyPolicy(learning_algo, env.get_num_action(), rng, 1.0),
        test_policy=EpsilonGreedyPolicy(learning_algo, env.get_num_action(), rng, 1.0),
    )

    # --- Create unique filename for FindBestController ---
    h = hash(args, hash_name="sha1")
    filename = "test_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(args))

    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(
        bc.EpsilonController(
            initial_e=train_args["epsilon_start"],
            e_decays=train_args["epsilon_decay"],
            e_min=train_args["epsilon_min"],
            evaluate_on="action",
            periodicity=1,
            reset_every="none",
        )
    )

    agent.run(n_epochs=10, epoch_length=500)
    print("end gathering data")

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(evaluate_on="epoch", periodicity=1))

    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(
        bc.LearningRateController(
            initial_learning_rate=train_args["learning_rate"],
            learning_rate_decay=train_args["learning_rate_decay"],
            periodicity=1,
        )
    )

    # Same for the discount factor.
    agent.attach(
        bc.DiscountFactorController(
            initial_discount_factor=train_args["discount"],
            discount_factor_growth=train_args["discount_inc"],
            discount_factor_max=train_args["discount_max"],
            periodicity=1,
        )
    )

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(
        bc.TrainerController(
            evaluate_on="action",
            periodicity=train_args["update_frequency"],
            show_episode_avg_v_value=True,
            show_avg_bellman_residual=True,
        )
    )

    # We wish to discover, among all versions of our neural network (i.e., after every training epoch), which one
    # seems to generalize the better, thus which one has the highest validation score. Here, we do not care about the
    # "true generalization score", or "test score".
    # To achieve this goal, one can use the FindBestController along with an InterleavedTestEpochControllers. It is
    # important that the validationID is the same than the id argument of the InterleavedTestEpochController.
    # The FindBestController will dump on disk the validation scores for each and every network, as well as the
    # structure of the neural network having the best validation score. These dumps can then used to plot the evolution
    # of the validation and test scores (see below) or simply recover the resulting neural network for your
    # application.
    agent.attach(
        bc.FindBestController(
            validation_id=MyEnv.VALIDATION_MODE,
            unique_fname=filename,
        )
    )

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a
    # "validation epoch" between each training epoch. For each validation epoch, we want also to display the sum of all
    # rewards obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env
    # every [parameters.period_btw_summary_perfs] *validation* epochs.
    agent.attach(
        bc.InterleavedTestEpochController(
            id=MyEnv.VALIDATION_MODE,
            epoch_length=experiment_args["steps_per_test"],
            periodicity=1,
            show_score=True,
            summarize_every=1,
        )
    )

    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass

    dump(args, "params/" + filename + ".jldump")

    agent.gathering_data = False
    agent.run(
        n_epochs=experiment_args["epochs"],
        epoch_length=experiment_args["steps_per_epoch"],
    )

    # --- Show results ---
    basename = "scores/" + filename
    scores = load(basename + "_scores.jldump")
    print(scores)
