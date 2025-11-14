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
import glob
import re
import numpy as np
import torch as th

def _find_last_checkpoints(log_folder, n=5):
    """Return a list of (timestep, path) for the last n checkpoints in log_folder."""
    pattern = os.path.join(log_folder, "model_*_steps.zip")
    files = glob.glob(pattern)
    items = []
    for f in files:
        m = re.search(r"model_(\d+)_steps\.zip$", f)
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda x: x[0])
    if len(items) == 0:
        return []
    return items[-n:]


if __name__ == "__main__":
    # Copy of result_ppo_lstm with a loop over recent checkpoints
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


    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy
    other_callbacks = []

    if args_baseline['save_video']:
        record_env_callback = PruningEvalRecordEnvCallback(verbose=args_callback['verbose'])
        other_callbacks.append(record_env_callback)

    # Determine logs folder from args
    load_path_arg = parsed_args_dict['args_global'].get('load_path')
    if not load_path_arg:
        print("ERROR: args_global.load_path is not set. Please pass --load_path to point to the logs subfolder.")
        sys.exit(1)

    logs_folder = os.path.join("./logs", load_path_arg)
    checkpoints = _find_last_checkpoints(logs_folder, n=5)
    print(f"Found {len(checkpoints)} checkpoints in {logs_folder}: {[t for t, p in checkpoints]}")
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {logs_folder}")
        sys.exit(1)

    results = []
    for timestep, path in checkpoints:
        print("Loading model at timestep: ", timestep, "from", path)
        load_timestep = timestep
        load_path_model = path

        logging_callback = PruningLogResultCallback(timestep=load_timestep, verbose=args_callback['verbose'])

        model = RecurrentPPOAE.load(load_path_model, env=env)
        # Move policy to device and set logger for consistent logging
        model.policy.to(utils.get_device())
        model.num_timesteps = load_timestep
        model._num_timesteps_at_start = load_timestep
        model.set_logger(new_logger)

        eval_method = GenerateResults(model, env, verbose=args_callback['verbose'], set_goal_callback=set_goal_callback,
                                      log_callback=logging_callback, other_callbacks=other_callbacks, type=type, name=load_path_arg)

        eval_method.run()

        # extract success list from logging_callback
        try:
            success_list = logging_callback._episode_info.get('is_success', None)
            if success_list is None:
                print("WARNING: No 'is_success' in logged episode info for timestep", load_timestep)
                success_rate = None
            else:
                # convert to numpy and compute mean (booleans to 0/1)
                success_rate = float(np.mean(np.array(success_list, dtype=float)))
        except Exception as e:
            print("Error extracting success rate:", e)
            success_rate = None

        results.append((load_timestep, success_rate))

        # cleanup
        del model
        if hasattr(th.cuda, 'empty_cache'):
            try:
                th.cuda.empty_cache()
            except Exception:
                pass

    print("Success rates for last checkpoints (timestep, success_rate):")
    for t, r in results:
        print(t, r)
