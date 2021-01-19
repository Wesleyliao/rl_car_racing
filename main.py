import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import cv2
from gym import wrappers

from agents.ddpg import DDPG

FRAMES_PER_ACTION = 10

def process_state(state_list, cur_state):
    
    grayscale = cv2.cvtColor(cur_state, cv2.COLOR_BGR2GRAY)
    state_list.append(grayscale)
    if len(state_list) > 5:
        state_list.pop(0)
    
    return np.array(state_list).T

# PERFORMING SINGLE SET
def perform_set(env, agent):

    total_rewards = []

    for i in range(1000):
        state_list = []
        _ = process_state(state_list, env.reset())
        done = False
        total_reward = 0 
        
        
        for j in range(10):
            s, _, _, _ = env.step([0,0,0])
            processed_s = process_state(state_list, s)

        j = 0

        while not done and j <= 1000:
            a = agent.pick_action(processed_s)

            r = 0
            for k in range(FRAMES_PER_ACTION):
                s_prime, reward, done, _ = env.step(a)
                r += reward
                env.render()

            total_reward += r
            processed_s_prime = process_state(state_list, s_prime)
            agent.update(processed_s, a, r, processed_s_prime, done)
            processed_s = processed_s_prime

            # print('iteration', i, 'frame', j)
            j += 1
            
        if i % 1 == 0: print('--- Iteration',i, 'total reward', total_reward) #, 'noise sigma', agent.sigma)
        total_rewards.append(total_reward)
    
    return np.array(total_rewards)

# FUNCTION FOR MAKING CHARTS
def make_charts(agent, mean, std):
    n = max(mean.shape)
    index = np.arange(n)
    plt.clf()
    plt.fill_between(index, mean - std, mean + std, alpha=0.2)
    plt.plot(index, mean, '-', linewidth=1, markersize=1, label=None)
    # plt.legend(title=legend_name, loc="best")
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')

    plt.title(agent.description)
    plt.savefig(os.path.join('images', agent.description + '.png'))


# MAIN
def main():

    # SET ENVIRONMENT
    # env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('LunarLander-v2')
    env = gym.make('CarRacing-v0')
    # env = wrap_env(env)
    # env = wrappers.Monitor(env, 'videos', video_callable=False ,force=True)

    # SET AGENT
    # agent = dqn.DQN(env)
    # agent = double_dqn.DoubleDQN(env)
    # agent = dueling_dqn.DuelingDQN(env)
    # agent = policy_gradient.PolicyGradient(env)
    # agent = ac.ActorCritic(env)
    # agent = a2c.AdvantageActorCritic(env)
    # agent = prioritized_ddqn.PrioritizedDDQN(env)
    agent = DDPG(env)
    # agent = a2c_continuous.AdvantageActorCriticContinuous(env)
    # agent = ppo.ProximalPolicyOptimization(env)

    all_run_rewards = []
    for i in range(1):
        print(f'### Starting Run {i+1} ###')
        agent.reset()
        all_run_rewards.append(perform_set(env, agent))

    all_run_rewards = np.array(all_run_rewards)
    mean = all_run_rewards.mean(axis=0)
    std = all_run_rewards.std(axis=0)
    make_charts(agent, mean, std)

if __name__ == '__main__':
    main()