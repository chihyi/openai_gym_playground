import gym
from gym import envs
#print envs.registry.all()

env = gym.make('FrozenLake-v0')
print env.action_space
print env.observation_space
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print "observation starts"
        print observation
        print "observation ends"
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
