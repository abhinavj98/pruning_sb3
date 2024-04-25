import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pruning_sb3.algo.PPOLSTMAE.policies import RecurrentActorCriticPolicy
from pruning_sb3.pruning_gym.custom_callbacks import CustomEvalCallback, CustomTrainCallback
from pruning_sb3.algo.PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from pruning_sb3.pruning_gym.pruning_env import PruningEnv
from pruning_sb3.pruning_gym.models import AutoEncoder
# import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse
# from args import args_dict
from pruning_sb3.args.args_test import \
    args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
import multiprocessing as mp
from pruning_sb3.pruning_gym.helpers import init_wandb
import copy

if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser()
    set_args(args, parser)
    parsed_args = vars(parser.parse_args())
    parsed_args_dict = organize_args(parsed_args)
    manager = mp.Manager()
    shared_list = manager.list()

    if parsed_args_dict['args_global']['load_path']:
        load_path = "../logs/run/best_model.zip"  # "./logs/{}/best_model.zip".format(args.LOAD_PATH)#./nfs/stak/users/jainab/hpc-share/codes/pruning_sb3/logs/lowlr/best_model.zip"#Nonei
    else:
        load_path = None

    # add_arg_to_env('shared_tree_list', shared_list, ['args_train'], parsed_args_dict)
    # Duplicates are resolved in favor of the value in x; dict(y, **x)
    args_global = parsed_args_dict['args_global']
    args_train = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_train'])
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    args_record = dict(args_test, **parsed_args_dict['args_record'])
    print(args_train)
    # Make an environment as usual
    data_env = PruningEnv(**args_train, make_trees=True)
    for i in data_env.trees:
        shared_list.append(copy.deepcopy(i))

    del data_env


    env = make_vec_env(PruningEnv, env_kwargs=args_train, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger
    eval_env = make_vec_env(PruningEnv, env_kwargs=args_test, vec_env_cls=SubprocVecEnv, n_envs=1)
    record_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=1)
    eval_env.logger = new_logger
    # Use deterministic actions for evaluation
    eval_callback = CustomEvalCallback(eval_env, record_env, best_model_save_path="../logs/test",
                                       log_path="../logs/test",
                                       deterministic=True, render=False, **parsed_args_dict['args_callback'])
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    # video_recorder = VideoRecorderCallback(eval_env, render_freq=1000)
    train_callback = CustomTrainCallback(trees=shared_list)


    policy_kwargs = {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": parsed_args_dict['args_policy']['state_dim'],
                                      "in_channels": 3, },
        "optimizer_class": th.optim.Adam,
        "log_std_init": parsed_args_dict['args_policy']['log_std_init'],
        "net_arch": dict(
            qf=[parsed_args_dict['args_policy']['emb_size'] * 2, parsed_args_dict['args_policy']['emb_size'],
                parsed_args_dict['args_policy']['emb_size'] // 2],
            pi=[parsed_args_dict['args_policy']['emb_size'] * 2, parsed_args_dict['args_policy']['emb_size'],
                parsed_args_dict['args_policy']['emb_size'] // 2]),
        "share_features_extractor": True,
        "n_lstm_layers": 2,
    }
    policy = RecurrentActorCriticPolicy

    # model = RecurrentPPOAE(policy, env, policy_kwargs = policy_kwargs, learning_rate = linear_schedule(args.LEARNING_RATE), learning_rate_ae=exp_schedule(args.LEARNING_RATE_AE), learning_rate_logstd = linear_schedule(0.01), n_steps=args.STEPS_PER_EPOCH, batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS)
    if load_path:
        model = RecurrentPPOAE.load(load_path, env)
        model.num_timesteps = 100000
        model._num_timesteps_at_start = 100000
        print(model.num_timesteps)
    else:
        model = RecurrentPPOAE(policy, env, policy_kwargs=policy_kwargs,
                               learning_rate=linear_schedule(parsed_args_dict['args_policy']['learning_rate']),
                               learning_rate_ae=exp_schedule(parsed_args_dict['args_policy']['learning_rate_ae']),
                               learning_rate_logstd=None,
                               n_steps=parsed_args_dict['args_policy']['steps_per_epoch'],
                               batch_size=parsed_args_dict['args_policy']['batch_size'],
                               n_epochs=parsed_args_dict['args_policy']['epochs'])

    model.set_logger(new_logger)
    train_callback.init_callback(model)
    print("Policy on device: ", model.policy.device)
    print("Model on device: ", model.device)
    print("Optical flow on device: ", model.policy.optical_flow_model.device)
    print("Using device: ", utils.get_device())

    model.learn(10000000, callback=[train_callback, eval_callback], progress_bar=False, reset_num_timesteps=False)
