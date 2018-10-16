import numpy
import gym
import time
"""
Args:
policy: [S, A] shaped matrix representing the policy.
env: OpenAI gym env.
env.P represents the transition probabilities of the environment.
env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
env.nS is a number of states in the environment.
env.nA is a number of actions in the environment.
gamma: Gamma discount factor.
render: boolean to turn rendering on/off.
"""
def execute(env, policy, gamma=1.0, render=False):
    start = env.reset()
    totalReward = 0
    stepIndex = 0
    while True:
        if render:
            env.render() 
        start, reward, done, _ = env.step(int(policy[start]))
        totalReward += (gamma ** stepIndex * reward)
        stepIndex += 1
        if done:
            break
    return totalReward

# Evaluates a policy by running it n times.returns:average total reward
def evaluatePolicy(env, policy, gamma=1.0, n=100):
    scores = [
        execute(env, policy, gamma=gamma, render=False)
        for _ in range(n)]
    return numpy.mean(scores)

# choosing the policy given a value-function
def calculatePolicy(v, gamma=1.0):
    policy = numpy.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = numpy.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
            # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = numpy.argmax(q_sa)
    return policy

# Value Iteration Agorithm
def valueIteration(env, gamma=1.0):
    value = numpy.zeros(env.env.nS) # initialize value-function
    max_iterations = 10000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = numpy.copy(value)
        for s in range(env.env.nS): 
            q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            value[s] = max(q_sa)
        if (numpy.sum(numpy.fabs(prev_v - value)) <= eps):
            print('Value-iteration converged at # %d.' % (i + 1))
            break
    return value

if __name__ == '__main__':
    gamma = 1.0
    env = gym.make("FrozenLake-v0")
    optimalValue = valueIteration(env, gamma);
    startTime = time.time()
    policy = calculatePolicy(optimalValue, gamma)
    policy_score = evaluatePolicy(env, policy, gamma, n=1000)
    endTime = time.time()
    print("Best score = %0.2f. Time taken = %4.4f seconds" % (numpy.mean(policy_score),endTime - startTime)) 