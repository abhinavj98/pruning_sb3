import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.pruning_gym.models import AutoEncoder

from pruning_sb3.pruning_gym.callbacks.eval_callbacks import PruningEvalSetGoalCallback, PruningEvalRecordEnvCallback, \
    PruningLogResultCallback, GenerateResults

from pruning_sb3.pruning_gym.pruning_env import PruningEnv

from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import argparse
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import set_args, organize_args, make_or_bins, \
    get_policy_kwargs
import pickle

if __name__ == "__main__":
    type = "uniform"
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    args_global, args_train, args_test, args_record, args_callback, args_policy, args_env, args_eval, args_baseline, parsed_args_dict = organize_args(
        parsed_args)
    verbose = 1

    load_timestep = args_global['load_timestep']

    if args_global['load_path']:
        load_path_model = "./logs/{}/model_{}_steps.zip".format(
            args_global['load_path'], load_timestep)
        load_path_mean_std = "./logs/{}/model_mean_std_{}_steps.pkl".format(
            args_global['load_path'], load_timestep)
    else:
        load_path_model = None

    print(parsed_args_dict)
    or_bins = make_or_bins(args_train, "test")

    env = make_vec_env(PruningEnv, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger

    dataset = None
    num_points_per_or = args_callback['n_points_per_orientation']
    num_orientations = args_callback['n_eval_orientations']
    if os.path.exists(f"{type}_dataset_{num_points_per_or}_{num_orientations}.pkl"):
        with open(f"{type}_dataset_{num_points_per_or}_{num_orientations}.pkl", "rb") as f:
            dataset = pickle.load(f)
    # if os.path.exists(f"{type}_dataset.pkl"):
    #     with open(f"{type}_dataset.pkl", "rb") as f:
    #         dataset = pickle.load(f)

    set_goal_callback = PruningEvalSetGoalCallback(or_bins=or_bins, type=type, dataset=dataset,
                                                   num_orientations=args_callback['n_eval_orientations'],
                                                   num_points_per_or=args_callback['n_points_per_orientation'],
                                                   verbose=args_callback['verbose'])
    record_env_callback = PruningEvalRecordEnvCallback(verbose=args_callback['verbose'])
    logging_callback = PruningLogResultCallback(verbose=args_callback['verbose'])

    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy

    model = RecurrentPPOAE.load(load_path_model, env=env)
    model.policy.load_running_mean_std_from_file(load_path_mean_std)
    model.num_timesteps = load_timestep
    model._num_timesteps_at_start = load_timestep
    model.set_logger(new_logger)

    eval_method = GenerateResults(model, env, verbose=args_callback['verbose'], set_goal_callback=set_goal_callback,
                                  log_callback=logging_callback)
    if verbose > 0:
        print("INFO: Policy on device: ", model.policy.device)
        print("INFO: Model on device: ", model.device)
        print("INFO: Optical flow on device: ", model.policy.optical_flow_model.device)
        print("INFO: Using device: ", utils.get_device())
        print("INFO: Number of timesteps: ", model.num_timesteps)

    eval_method.run()
