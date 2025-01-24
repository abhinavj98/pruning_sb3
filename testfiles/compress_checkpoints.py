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

    print(parsed_args_dict)
    or_bins = make_or_bins(args_test, "test", args_global['tree_type'])

    env = make_vec_env(PruningEnv, env_kwargs=args_record, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger

    dataset = None
    num_points_per_or = args_callback['n_points_per_orientation']
    num_orientations = args_callback['n_eval_orientations']
    if os.path.exists(f"{type}_dataset_{num_points_per_or}_{num_orientations}.pkl"):
        with open(f"{type}_dataset_{num_points_per_or}_{num_orientations}.pkl", "rb") as f:
            dataset = pickle.load(f)

    set_goal_callback = PruningEvalSetGoalCallback(or_bins=or_bins, type=type, dataset=dataset,
                                                   num_orientations=args_callback['n_eval_orientations'],
                                                   num_points_per_or=args_callback['n_points_per_orientation'],
                                                   verbose=args_callback['verbose'])
    record_env_callback = PruningEvalRecordEnvCallback(verbose=args_callback['verbose'])


    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy

    load_timestep_list = [3224000, 3348000, 3472000]
    for i in range(len(load_timestep_list)):
        load_timestep = load_timestep_list[i]
        logging_callback = PruningLogResultCallback(timestep = load_timestep, verbose=args_callback['verbose'])
        print("Loading model at timestep: ", load_timestep)
        if parsed_args_dict['args_global']['load_path']:
            load_path_model = "./logs/{}/model_{}_steps.zip".format(
                parsed_args_dict['args_global']['load_path'], load_timestep)

            save_path_model = "./logs/{}/model_{}_steps.zip".format(
                parsed_args_dict['args_global']['load_path']+'_compressed', load_timestep)

        model = RecurrentPPOAE.load(load_path_model, env=env)

        model.save(f"{save_path_model}/model_{load_timestep}_steps.zip", exclude=["_last_obs", "_last_episode_starts", "_last_original_obs",
                                    "_last_obs", "_last_episode_starts", "_last_original_obs",
                                    "ep_info_buffer", "ep_success_buffer", "_last_obs_expert",
                                    "_last_lstm_states_expert", "rollout_buffer","expert_buffer"])
