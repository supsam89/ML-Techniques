import numpy
import time
import gym
"""
Args:
policy: [S, A] shaped matrix representing the policy.
env: OpenAI gym env.
render: boolean to turn rendering on/off.
"""
#Execution
def execute(env, policy, episodeLength=100, render=False):
    totalReward = 0
    start = env.reset()
    for t in range(episodeLength):
        if render:
            env.render()
        action = policy[start]
        start, reward, done, _ = env.step(action)
        totalReward += reward
        if done:
            break
    return totalReward

#Evaluation
def evaluatePolicy(env, policy, n_episodes=100):
    totalReward = 0.0
    for _ in range(n_episodes):
        totalReward += execute(env, policy)
    return totalReward / n_episodes

#Function for a random policy
def gen_random_policy():
    return numpy.random.choice(4, size=((16))) 

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    ## Policy search
    n_policies = 1000
    startTime = time.time()
    policy_set = [gen_random_policy() for _ in range(n_policies)]
    policy_score = [evaluatePolicy(env, p) for p in policy_set]
    endTime = time.time()
    print("Best score = %0.2f. Time taken = %4.4f seconds" %(numpy.max(policy_score) ,
    endTime - startTime)) 