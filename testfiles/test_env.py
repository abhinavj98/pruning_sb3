from stable_baselines3.common.env_checker import check_env
from gym_env_discrete import PruningEnv
#from stable_baselines3.common.vec_env.base_vec_env import DummyVecEnv

env = PruningEnv()
#env = DummyVecEnv(env)
# It will check your custom environment and output additional warnings if needed
check_env(env)
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
#model = A2C('CnnPolicy', env).learn(total_timesteps=1000)
