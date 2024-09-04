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
from stable_baselines3.common.evaluation import evaluate_policy
import glob
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
    # or_bins = make_or_bins(args_train, "test")

    env = PruningEnv(**args_record)
    # new_logger = utils.configure_logger(verbose=0, tensorboard_log="./runs/", reset_num_timesteps=True)
    # env.logger = new_logger

    #load pkl file from expert_trajectories_temp
    expert_trajectory_path = "expert_trajectories_temp"
    expert_trajectories = glob.glob(expert_trajectory_path + "/*.pkl")

    policy_kwargs = get_policy_kwargs(args_policy, args_env, AutoEncoder)
    policy = RecurrentActorCriticPolicy

    model = RecurrentPPOAE.load(load_path_model, env=env)
    model.policy.load_running_mean_std_from_file(load_path_mean_std)
    model.num_timesteps = load_timestep
    model._num_timesteps_at_start = load_timestep

    #shuffle the expert trajectories
    # random.shuffle(expert_trajectories)
    for expert_trajectory in expert_trajectories[:50]:
        print("Expert trajectory: ", expert_trajectory)
        with open(expert_trajectory, "rb") as f:
            expert_data = pickle.load(f)
        print("Expert data: ", expert_data['actions'])
        tree_info = expert_data['tree_info']
        actions = expert_data['actions']
        observations = expert_data['observations']
        # dones = expert_data['dones']
        env.set_tree_properties(*tree_info)

        env.ur5.reset_ur5_arm()
        env.reset()
        for i in range(len(actions)):
            action = actions[i]
            print("Action: ", action)
            observation = observations[i]
            # env.set_observation(observation)
            # env.set_action(action)
            obs, rew, term, trunc, _ = env.step(action)
            print(rew, term)

    # env.reset()



        env.reset()

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=1,
            render=False,
            deterministic=True,
            return_episode_rewards=True
        )

        print("Episode rewards: ", episode_rewards)
        if verbose > 0:
            print("INFO: Policy on device: ", model.policy.device)
            print("INFO: Model on device: ", model.device)
            print("INFO: Optical flow on device: ", model.policy.optical_flow_model.device)
            print("INFO: Using device: ", utils.get_device())
            print("INFO: Number of timesteps: ", model.num_timesteps)

