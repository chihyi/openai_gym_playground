import gym
from gym import envs
print envs.registry.all()

#env = gym.make('CartPole-v0')
#for i_episode in range(20):
#    observation = env.reset()
#    for t in range(100):
#        env.render()
#        print(env.action_space.sample())
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(0)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
