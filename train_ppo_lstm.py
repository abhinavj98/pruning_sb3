from PPOLSTMAE.policies import RecurrentActorCriticPolicy
from custom_callbacks import CustomEvalCallback, CustomTrainCallback
from PPOLSTMAE.ppo_recurrent_ae import RecurrentPPOAE
from gym_env_discrete import PruningEnv
from models import AutoEncoder
# import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import argparse
from args import args_dict
# from args_test import args_dict

from helpers import init_wandb, linear_schedule, exp_schedule, optical_flow_create_shared_vars

# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser based on the dictionary
for arg_name, arg_params in args_dict.items():
    parser.add_argument(f'--{arg_name}', **arg_params)

# Parse arguments from the command line
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    init_wandb(args)
    # TODO: put in args
    optical_flow_subproc = True
    if args.USE_OPTICAL_FLOW and optical_flow_subproc:
        print("Using optical flow")
        shared_var = optical_flow_create_shared_vars()
    else:
        shared_var = (None, None)

    if args.LOAD_PATH:

        load_path = "./logs/{}/best_model.zip".format(
            args.LOAD_PATH)  # ./nfs/stak/users/jainab/hpc-share/codes/pruning_sb3/logs/lowlr/best_model.zip"#Nonei
    else:
        load_path = None
    train_env_kwargs = {"renders": args.RENDER, "tree_urdf_path": args.TREE_TRAIN_URDF_PATH,
                        "tree_obj_path": args.TREE_TRAIN_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                        "max_steps": args.MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                        "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                        "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                        "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                        "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                        "slack_reward_scale": args.SLACK_REWARD_SCALE,
                        "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                        "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                        "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                        "shared_var": shared_var}

    eval_env_kwargs = {"renders": False, "tree_urdf_path": args.TREE_TEST_URDF_PATH,
                       "tree_obj_path": args.TREE_TEST_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                       "max_steps": args.EVAL_MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                       "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                       "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                       "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                       "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                       "slack_reward_scale": args.SLACK_REWARD_SCALE, "num_points": args.EVAL_POINTS,
                       "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                       "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                       "name": "evalenv", "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                       "shared_var": shared_var}

    record_env_kwargs = {"renders": False, "tree_urdf_path": args.TREE_TEST_URDF_PATH,
                         "tree_obj_path": args.TREE_TEST_OBJ_PATH, "action_dim": args.ACTION_DIM_ACTOR,
                         "max_steps": args.EVAL_MAX_STEPS, "movement_reward_scale": args.MOVEMENT_REWARD_SCALE,
                         "action_scale": args.ACTION_SCALE, "distance_reward_scale": args.DISTANCE_REWARD_SCALE,
                         "condition_reward_scale": args.CONDITION_REWARD_SCALE,
                         "terminate_reward_scale": args.TERMINATE_REWARD_SCALE,
                         "collision_reward_scale": args.COLLISION_REWARD_SCALE,
                         "slack_reward_scale": args.SLACK_REWARD_SCALE, "num_points": args.EVAL_POINTS,
                         "pointing_orientation_reward_scale": args.POINTING_ORIENTATION_REWARD_SCALE,
                         "perpendicular_orientation_reward_scale": args.PERPENDICULAR_ORIENTATION_REWARD_SCALE,
                         "name": "recordenv", "use_optical_flow": args.USE_OPTICAL_FLOW, "optical_flow_subproc": True,
                         "shared_var": shared_var}

    env = make_vec_env(PruningEnv, env_kwargs=train_env_kwargs, n_envs=args.N_ENVS, vec_env_cls=SubprocVecEnv)

    new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    env.logger = new_logger
    eval_env = make_vec_env(PruningEnv, env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, n_envs=1)  # args.N_ENVS)

    record_env = make_vec_env(PruningEnv, env_kwargs=record_env_kwargs, vec_env_cls=SubprocVecEnv, n_envs=1)
    eval_env.logger = new_logger
    # Use deterministic actions for evaluation
    eval_callback = CustomEvalCallback(eval_env, record_env, best_model_save_path="./logs/{}".format(args.NAME),
                                       log_path="./logs/{}".format(args.NAME), eval_freq=args.EVAL_FREQ,
                                       deterministic=True, render=False, n_eval_episodes=args.EVAL_EPISODES)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    train_callback = CustomTrainCallback()

    policy_kwargs = {
        "features_extractor_class": AutoEncoder,
        "features_extractor_kwargs": {"features_dim": args.STATE_DIM,
                                      "in_channels": (3 if args.USE_OPTICAL_FLOW else 1), },
        "optimizer_class": th.optim.Adam,
        "log_std_init": args.LOG_STD_INIT,
        "net_arch": dict(qf=[args.EMB_SIZE * 2, args.EMB_SIZE], pi=[args.EMB_SIZE * 2, args.EMB_SIZE]),
        "share_features_extractor": False,
        "n_lstm_layers": 1,
        "features_dim_critic_add": 2,
        "squash_output": True,  # Doesn't work
    }
    policy = RecurrentActorCriticPolicy

    if not load_path:
        model = RecurrentPPOAE(policy, env, use_sde=False, policy_kwargs=policy_kwargs,
                               learning_rate=linear_schedule(args.LEARNING_RATE),
                               learning_rate_ae=exp_schedule(args.LEARNING_RATE_AE),
                               learning_rate_logstd=linear_schedule(0.01), n_steps=args.STEPS_PER_EPOCH,
                               batch_size=args.BATCH_SIZE, n_epochs=args.EPOCHS,)
    else:
        load_dict = {"learning_rate": linear_schedule(args.LEARNING_RATE),
                     "learning_rate_ae": exp_schedule(args.LEARNING_RATE_AE),
                     "learning_rate_logstd": linear_schedule(0.01)}
        model = RecurrentPPOAE.load(load_path, env=env, custom_objects=load_dict)
        model.num_timesteps = 10_000_000
        model._num_timesteps_at_start = 10_000_000
        print("LOADED MODEL")
    model.set_logger(new_logger)
    print("Using device: ", utils.get_device())

    # env.reset()
    model.learn(20_000_000, callback=[train_callback, eval_callback], progress_bar=False, reset_num_timesteps=False)
