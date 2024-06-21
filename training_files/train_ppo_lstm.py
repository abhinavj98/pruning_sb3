import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.pruning_gym.models import AutoEncoder

from pruning_sb3.pruning_gym.callbacks.callbacks import EveryNRollouts, PruningLogCallback
from pruning_sb3.pruning_gym.callbacks.train_callbacks import PruningTrainSetGoalCallback, \
    PruningTrainRecordEnvCallback, PruningCheckpointCallback

from pruning_sb3.pruning_gym.pruning_env import PruningEnv

from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import argparse
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import linear_schedule, set_args, organize_args, init_wandb, make_or_bins, \
    get_policy_kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, parsed_args_dict = organize_args(
        parsed_args)
    verbose = 1

    init_wandb(parsed_args_dict, args_global['run_name'])
    load_timestep = args_global['load_timestep']

    if args_global['load_path']:
        load_path_model = "./logs/{}/model_{}_steps.zip".format(
            args_global['load_path'], load_timestep)
        load_path_mean_std = "./logs/{}/model_mean_std_{}_steps.pkl".format(
            args_global['load_path'], load_timestep)
    else:
        load_path_model = None

    print(parsed_args_dict)
    or_bins = make_or_bins(args_train, "train")

    env = make_vec_env(PruningEnv, env_kwargs=args_train, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger

    set_goal_callback = PruningTrainSetGoalCallback(or_bins=or_bins, verbose=args_callback['verbose'])
    checkpoint_callback = PruningCheckpointCallback(save_freq=args_callback['save_freq'],
                                                    save_path="./logs/{}".format(args_global['run_name']),
                                                    name_prefix="model", verbose=args_callback['verbose'])
    record_env_callback = EveryNRollouts(200, PruningTrainRecordEnvCallback(verbose=args_callback['verbose']))
    logging_callback = PruningLogCallback(verbose=args_callback['verbose'])
    callback_list = [record_env_callback, set_goal_callback, checkpoint_callback, logging_callback]

    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy

    if not load_path_model:
        model = RecurrentPPOAE(policy, env, policy_kwargs=policy_kwargs,
                               learning_rate=linear_schedule(args_policy['learning_rate']),
                               learning_rate_ae=linear_schedule(args_policy['learning_rate_ae']),
                               learning_rate_logstd=None,
                               n_steps=args_policy['steps_per_epoch'],
                               batch_size=args_policy['batch_size'],
                               n_epochs=args_policy['epochs'],
                               ae_coeff=args_policy['ae_coeff'])
    else:
        load_dict = {"learning_rate": linear_schedule(args_policy['learning_rate']),
                     "learning_rate_ae": linear_schedule(args_policy['learning_rate_ae']),
                     "learning_rate_logstd": None,
                     "log_std": -0.5}
        model = RecurrentPPOAE.load(load_path_model, env=env, custom_objects=load_dict)

        model.policy.load_running_mean_std_from_file(load_path_mean_std)

        model.num_timesteps = load_timestep
        model._num_timesteps_at_start = load_timestep
        print("INFO: Loaded Model")

    model.set_logger(new_logger)
    set_goal_callback.init_callback(model)

    if verbose > 0:
        print("INFO: Policy on device: ", model.policy.device)
        print("INFO: Model on device: ", model.device)
        print("INFO: Optical flow on device: ", model.policy.optical_flow_model.device)
        print("INFO: Using device: ", utils.get_device())

    model.learn(args_policy['total_timesteps'], callback=callback_list, progress_bar=False, reset_num_timesteps=False)
