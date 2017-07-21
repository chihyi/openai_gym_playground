import gym
from gym import wrappers
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

upload_folder = "/tmp/cartpole-experiment-2"
LR = 1e-3
env = gym.make('FrozenLake-v0')
env.reset()
goal_steps = 300
score_requirement = 1
initial_games = 1000000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for _ in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        first_step_flag = True
        for _ in range(goal_steps):
            action = env.action_space.sample()   
            observation, reward, done, info = env.step(action)
            if first_step_flag == False:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            first_step_flag = False
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0:
                    output = [1, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1]
            
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    #training_data_save = np.array(training_data)
    #np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data
            
def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(len(training_data), 1, 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = 1)

    model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model

def test_model(model):
    scores = []
    choices = []
    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        first_step_flag = True
        for _ in range(goal_steps):
            #env.render()
            if first_step_flag == True:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict([[[prev_obs]]])[0])
            first_step_flag = False
            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    return {'scores': scores, 'choices': choices}


training_data = initial_population()
model = train_model(training_data)

result = test_model(model)
#env = wrappers.Monitor(env, upload_folder)

scores = result['scores']
choices = result['choices']
print('Average Score', sum(scores)/len(scores))
print('Choice 0: {}, Choice 1: {}, Choice 2: {}, Choice 3: {}'.format(float(choices.count(0))/len(choices), float(choices.count(1))/len(choices), float(choices.count(2))/len(choices), float(choices.count(3))/len(choices)))

#env.close()
#gym.upload(upload_folder, api_key='sk_4mEnOlgJQeKLv6djQnww')


