import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class GaussianPolicyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GaussianPolicyNN, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc2_mean = nn.Linear(hidden_size, output_size)  # Output for mean
        # self.fc2_std = nn.Linear(hidden_size, output_size)  # Output for std
        self.mean = torch.nn.Parameter(torch.tensor([.0]))
        self.std = torch.nn.Parameter(torch.tensor([torch.log(torch.tensor(1))]))
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # # x = torch.relu(self.fc2(x))
        # mean = self.fc2_mean(x)
        # std = torch.exp(self.fc2_std(x))  # Ensure std is positive
        return self.mean, torch.exp(self.std)


def select_action(policy_net, state, detach_tanh):
    mean, std = policy_net(state)
    std = std
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    if detach_tanh:
        squashed_action = torch.tanh(action)
    else:
        noise = (action - mean.detach()) / std.detach()
        squashed_action = torch.tanh(mean + std * noise)
        assert torch.isclose(action, mean + std * noise).all()
        #Correction of log_prob due to squashing
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-8)

    return squashed_action.detach(), log_prob  # Return action and log probability


def compute_loss(log_probs, rewards):
    # Compute the policy gradient loss
    return -torch.sum(log_probs * rewards)

def train(target, detach_tanh = False):
    # Hyperparameters
    input_size = 1  # Dimension of the state
    hidden_size = 3  # Hidden layer size
    output_size = 1  # Action dimension
    num_episodes = 5000
    learning_rate = 0.01
    target_value = target  # Example target value

    # Create the policy network and optimizer
    policy_net = GaussianPolicyNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    continue_training = True
    total_episodes = 0
    # Training Loop
    for episode in range(num_episodes):
        if continue_training == False:
            break
        state = torch.tensor([[0.]], dtype=torch.float32)  # Initial state
        log_probs = []
        rewards = []

        for t in range(10):  # Run for 10 time steps
            action, log_prob = select_action(policy_net, state, detach_tanh)
            action = action.detach()


            # Calculate the predicted value (the action itself)
            predicted_value = action.item()
            # print(predicted_value)

            # Reward is the negative linear difference from the target value
            reward = -abs(predicted_value-target_value)#1/abs(predicted_value - target_value)*1/1000 #maximize this
            # Store log probability and reward
            # print(log_prob)
            log_probs.append(log_prob)

            rewards.append(reward)

            # Set next state (for this example, just use the action as the next state)
            # state = action.unsqueeze(0)  # Simple state update

        # Convert rewards to tensor and calculate cumulative rewards
        rewards = torch.tensor(rewards, dtype=torch.float64)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Normalize rewards
        # Compute loss and update policy
        loss = compute_loss(torch.cat(log_probs), rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_episodes += 1
        with torch.no_grad():
            mean, std = policy_net(state)
            action = torch.tanh(mean)
            # print(action - target_value)
            if abs(action - target_value) < 0.001:
                print(f"Target reached {abs(action - target_value)}")
                continue_training = False
            if (episode + 1) % 100 == 0:

                print(f"Rewards: {rewards.mean()}")


                print(f'Episode [{episode + 1}/{num_episodes}], Predicted mean: {action, mean.item()}, std: {std.item():.4f}')
                print(f'Episode [{episode + 1}/{num_episodes}], Loss: {loss.item():.4f}')

    print(f"Total episodes: {total_episodes}")
    print(mean.item(), torch.tanh(mean), std.item())
    return total_episodes

if __name__ == '__main__':
    #Run 100 episodes with detach_tanh = False
    target_list = [0.5]
    for target in target_list:
        epochs_false = []
        epochs_true = []
        for _ in range(100):
            epochs = train(target, False)
            print(epochs)
            epochs_false.append(epochs)
            # input()
        # Run 100 episodes with detach_tanh = True
        for _ in range(100):
            epochs = train(target, True)
            print(epochs)
            epochs_true.append(epochs)


        print(f"Average epochs for detach_tanh = False: {np.mean(np.array(epochs_false)), np.std(np.array(epochs_false))}")
        print(f"Average epochs for detach_tanh = True: {np.mean(np.array(epochs_true)), np.std(np.array(epochs_true))}")
        #Save the results to a file
        with open("tanh_results.txt", "a") as f:
            f.write(f"Target: {target}\n")
            f.write(f"Average epochs for detach_tanh = False: {np.mean(np.array(epochs_false)), np.std(np.array(epochs_false))}\n")
            f.write(f"Average epochs for detach_tanh = True: {np.mean(np.array(epochs_true)), np.std(np.array(epochs_true))}\n")
            f.write("\n")