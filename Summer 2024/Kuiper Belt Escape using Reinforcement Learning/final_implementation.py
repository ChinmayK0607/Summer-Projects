from collections import defaultdict
import gym
import math
import numpy as np
import random
import time
import gym_kuiper_escape 
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')

env = gym.make('kuiper-escape-base-v0',mode='None',rock_rate = 0.4,player_speed = 0.5,rock_size_min = 0.08,rock_size_max = 0.12, )

max_steps=1000 #max_steps in an episode

#defining parameters
initial_alpha=0.4
min_alpha=0.1
gamma=0.9
numOfepisodes=5000
min_epsilon=1e-2
max_epsilon=1.0
decay_rate = (max_epsilon-min_epsilon)/numOfepisodes #epsilon decay rate ( for linear decay )

#Q table initialization
def default_q_value():
    return np.zeros(5)
Q=defaultdict(default_q_value) # as new states are visited , its each action's value will be initialized as 0.
policy={}

# learning rate decay
def linear_decay_alpha(episode_num, initial_alpha, min_alpha, num_episodes):
    return max(min_alpha, initial_alpha - (initial_alpha - min_alpha) * (episode_num / num_episodes))   

# initializing parameters  
epi_count=0
episodic_rewards=[]
steps_per_epi=[]
epsilon=max_epsilon
alpha=initial_alpha

while True:
    state=env.reset() 
    episode_history=[]
    steps=0
    total_reward=0
    ''' Number of lidar beams was set to 8 on env_base file , it will give 16x1 array and
        since only first half gives collision distances it is further sliced to get 8x1 and then is flatten in tuple.
    '''
    state=tuple(state[:8].flatten()) 
    epi_count+=1
    while True:
        #---choosing action by epsilon greedy
        p=random.random()
        if p<=epsilon:
            action=random.randint(0,4)
        else:
            action= np.argmax(Q[state])
    
        next_obs, reward, done,info = env.step(action)
        steps+=1
        next_state=tuple(next_obs[:8].flatten())
        
        ##---Reward Modification:
        if done: 
            reward -=10 #panality for hitting the asteroid.
        ##-----
        episode_history.append(((state,action),reward,next_state,done)) # storing episodic data
        Q[state][action]+= alpha*(reward + gamma*(np.max(Q[next_state])-Q[state][action])) #Bellman otimality equation

        total_reward+=reward # accumulating total reward
        state=next_state
        
        if done or steps>=max_steps: #for episode termination
            break

    epsilon= max(min_epsilon,max_epsilon-(epi_count*decay_rate)) #epsilon decay
    alpha = linear_decay_alpha(epi_count, initial_alpha, min_alpha, numOfepisodes) # alpha decay
    
    if epi_count % 100 == 0:
        print(f" Total reward: {total_reward}, Epsilon: {epsilon},Episode: {epi_count}") #to get terminal feedback of learning process

    episodic_rewards.append(total_reward)
    steps_per_epi.append(steps)
    #policy formation
    for s in Q:
        policy[s] = np.argmax(Q[s])
    if epi_count>=numOfepisodes: #temination condition
        break

epi=np.arange(1,len(steps_per_epi)+1) #total number of episodes

#for plotting simple moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
window_size = 90

#-------Plotting Results-----------------
moving_avg = moving_average(episodic_rewards, window_size) 
plt.figure(dpi=200)
plt.plot(epi,episodic_rewards)
plt.title("Rewards/episode")
plt.xlabel("Episodes")
plt.ylabel("Rewards")

plt.suptitle("Training")
plt.plot(epi[window_size-1:], moving_avg, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
plt.legend()
plt.savefig('rewardVepisodes.png')
plt.show()
