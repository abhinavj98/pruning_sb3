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
from pruning_sb3.args.args import \
    args
from pruning_sb3.pruning_gym.helpers import linear_schedule, exp_schedule, optical_flow_create_shared_vars, \
    set_args, organize_args, add_arg_to_env, init_wandb

# Add arguments to the parser based on the dictionary
parser = argparse.ArgumentParser()
set_args(args, parser)
parsed_args = vars(parser.parse_args())
parsed_args_dict = organize_args(parsed_args)
print(parsed_args_dict)

if __name__ == "__main__":
    # init_wandb(parsed_args_dict, parsed_args_dict['args_global']['run_name'])
    if make_dataset:
        # Make an environment as usual
        data_env = PruningEnv(**parsed_args_dict['args_env'])

        # Put trees and curriculim points in shared memory
        # Pass the shared memory to training env

    if parsed_args_dict['args_global']['load_path']:
        load_path_model = "./logs/{}/current_model.zip".format(
            parsed_args_dict['args_global']['load_path'])
        load_path_mean_std = "./logs/{}/current_mean_std.pkl".format(
            parsed_args_dict['args_global']['load_path'])
    else:
        load_path_model = None

    args_global = parsed_args_dict['args_global']
    args_train = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_train'])
    args_test = dict(parsed_args_dict['args_env'], **parsed_args_dict['args_test'])
    args_record = dict(args_test, **parsed_args_dict['args_record'])
    args_policy = parsed_args_dict['args_policy']

    env = make_vec_env(PruningEnv, env_kwargs=args_train, n_envs=args_global['n_envs'], vec_env_cls=SubprocVecEnv)
    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger
    eval_env = make_vec_env(PruningEnv, env_kwargs=args_test, vec_env_cls=SubprocVecEnv, n_envs=1)
    record_env = make_vec_env(PruningEnv, env_kwargs=args_record, vec_env_cls=SubprocVecEnv, n_envs=1)
    eval_env.logger = new_logger
    # Use deterministic actions for evaluation
    eval_callback = CustomEvalCallback(eval_env, record_env, best_model_save_path="./logs/{}".format(args_global['run_name']),
                                       log_path="./logs/{}".format(args_global['run_name']),
                                       deterministic=True, render=False, **parsed_args_dict['args_callback'])
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    train_callback = CustomTrainCallback()

    policy_kwargs = {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": parsed_args_dict['args_policy']['state_dim'],
                                      "in_channels": 3,
                                      "size": (224, 224)},
        "optimizer_class": th.optim.Adam,
        "log_std_init": parsed_args_dict['args_policy']['log_std_init'],
        "net_arch": dict(
            qf=[args_policy['emb_size'] * 2, args_policy['emb_size'],
                args_policy['emb_size'] // 2],
            pi=[args_policy['emb_size'] * 2, args_policy['emb_size'],
                args_policy['emb_size'] // 2]),
        "activation_fn": th.nn.ReLU,
        "share_features_extractor": False,
        "n_lstm_layers": 1,
        "features_dim_critic_add": 2, #Assymetric critic
        # "squash_output": True,  # Doesn't work
    }
    policy = RecurrentActorCriticPolicy

    if not load_path_model:
        model = RecurrentPPOAE(policy, env, policy_kwargs=policy_kwargs,
                               learning_rate=linear_schedule(args_policy['learning_rate']),
                               learning_rate_ae=exp_schedule(args_policy['learning_rate_ae']),
                               learning_rate_logstd=linear_schedule(0.01),
                               n_steps=args_policy['steps_per_epoch'],
                               batch_size=args_policy['batch_size'],
                               n_epochs=args_policy['epochs'])
    else:
        load_dict = {"learning_rate": linear_schedule(args_policy['learning_rate']),
                     "learning_rate_ae": exp_schedule(args_policy['learning_rate_ae']),
                     "learning_rate_logstd": linear_schedule(0.01)}
        model = RecurrentPPOAE.load(load_path_model, env=env, custom_objects=load_dict)
        
        model.policy.load_running_mean_std_from_file(load_path_mean_std)
        model.num_timesteps = 2_500_000
        model._num_timesteps_at_start = 2_500_000
        print("LOADED MODEL")
    model.set_logger(new_logger)

    print("Policy on device: ", model.policy.device)
    print("Model on device: ", model.device)
    print("Optical flow on device: ", model.policy.optical_flow_model.device)
    print("Using device: ", utils.get_device())

    # env.reset()
    model.learn(args_policy['total_timesteps'], callback=[train_callback, eval_callback], progress_bar=False, reset_num_timesteps=False)
