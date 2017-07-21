import gym
from gym import wrappers
import numpy as np

np.random.seed(0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def weighed_avg(arr, weights):
	"""return weighted avg of reward history"""
	res = np.zeros(np.shape(arr[0]))

	soft_weights = softmax(weights)

	for i in range(len(arr)):
		res += arr[i] * soft_weights[i]

	return res


def run_episode(env, parameters):
	observation = env.reset()
	totalreward = 0
	time_steps = 0
	done = False

	while not done:
		env.render()
		#initalize random weights
		action = 0 if np.matmul(parameters, observation)<0 else 1
		observation, reward, done, info = env.step(action)
		totalreward += reward
		time_steps += 1

	return (totalreward, time_steps)


#training
def train(submit):
	env = gym.make('CartPole-v0')
	env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

	episodes_per_update = 5
	noise_scaling = 0.3
	parameters = np.random.rand(4) * 2 -1
	bestreward = 0
	total_steps = 500

	hist_length = 10
	reward_hist = []
	param_hist = []



	for step in range(total_steps):
		decay = np.exp(-step/total_steps)
		newparams = parameters + (np.random.rand(4) *2 -1) * noise_scaling * decay

		reward, time_steps = run_episode(env, newparams)

		print("Epoch : {step+1}, {time_steps} steps, Average : {bestreward}")

		# keep track of last `n` histories
		reward_hist.append(reward)
		reward_hist = reward_hist[-hist_length:]
		param_hist.append(newparams)
		param_hist = param_hist[-hist_length:]

		
		if reward > bestreward and step < 10:
			bestreward = reward

		if reward > bestreward and step >= 10: 
			bestreward = np.average(reward_hist)
			# average of last `n` params 
			parameters = weighed_avg(param_hist, reward_hist)
			
			# if reward == 200:
			# 	break

	env.close()

	if submit:
		gym.upload('/tmp/cartpole-experiment-1')

	return parameters

r = train(submit=False)

print("Trained params : {r}")
