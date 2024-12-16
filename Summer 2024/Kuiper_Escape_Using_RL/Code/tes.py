import gym_kuiper_escape
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import math
import matplotlib.pyplot as plt

# Parameters
num_episodes = 13001 
alpha = 0.6        
gamma = 0.7      
epsilon = 1.0      
epsilon_decay = 0.99  
min_epsilon = 0.0000001 
testing_start_episode = 10000

# Environment setup
env = gym.make('kuiper-escape-base-v0', mode='None', rock_rate=0.2, player_speed=0.5, rock_size_min=0.08, rock_size_max=0.1)
Q = {}
epi_num = []
epi_returns = []
epi_tes = []
tes_num = []

# Epsilon-greedy policy function
def epsilon_greedy_policy(state, epsilon):
    state = tuple(state)
    if np.random.rand() < epsilon:
        return np.random.randint(0, 5) 
    else:
        return np.argmax(Q.get(state, np.zeros(5))) 

for episode in range(num_episodes):
    obs = env.reset()
    obs = obs.flatten() 
    m = obs[:8]
    m1 = tuple(m)
    done = False
    total_reward = 0
    c = 0

    if episode >= testing_start_episode:
        # Testing phase (fixed epsilon)
        epsilon_test = 0.0  # Purely greedy during testing
        while not done and c <= 1000:
            action = epsilon_greedy_policy(obs, epsilon_test)
            next_obs, rew, done, _ = env.step(action)
            next_obs = next_obs.flatten()
            h = next_obs[:8]
            h1 = tuple(h)
            if done:
                rew += -10
            obs = next_obs 
            total_reward += rew
            c += 1

        epi_tes.append(total_reward)
        tes_num.append(episode)
        print(f"Testing Episode: {episode}, Rewards: {total_reward}")

    else:
        # Training phase
        while not done and c <= 1000:
            action = epsilon_greedy_policy(obs, epsilon)
            next_obs, rew, done, _ = env.step(action)
            next_obs = next_obs.flatten()
            h = next_obs[:8]
            h1 = tuple(h)
            if done:
                rew += -10  

            if m1 not in Q:
                Q[m1] = np.zeros(5)  

            if h1 not in Q:
                Q[h1] = np.zeros(5)

            Q[m1][action] += alpha * (rew + gamma * np.max(Q[h1]) - Q[m1][action])
            
            obs = next_obs 
            total_reward += rew
            c += 1

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        epi_returns.append(total_reward)
        print(f"Training Episode: {episode}, Rewards: {total_reward}")

env.close()

def calculate_sma(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

sma_window_size = 10
sma_rewards = calculate_sma(epi_returns, sma_window_size)
sma_testing_rewards = calculate_sma(epi_tes, sma_window_size)

# plt.figure(figsize=(10, 6), dpi=200)
# plt.plot(epi_returns, label='Reward per Episode (Training)', color='blue', alpha=0.6)
# plt.plot(range(sma_window_size - 1, len(epi_returns)), sma_rewards, label=f'{sma_window_size}-Episode SMA (Training)', color='red', linewidth=2)

plt.plot(tes_num, epi_tes, label='Reward per Episode (Testing)', color='green', alpha=0.6)
plt.plot(range(tes_num[0] + sma_window_size - 1, len(epi_tes) + tes_num[0]), sma_testing_rewards, label=f'{sma_window_size}-Episode SMA (Testing)', color='purple', linewidth=2)

plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episodes with SMA (Testing)')
plt.grid(True)
plt.legend()
plt.savefig('reward_vs_episodes_with_sma3.png')
print("Graph with SMA saved as 'reward_vs_episodes_with_sma1.png'")
