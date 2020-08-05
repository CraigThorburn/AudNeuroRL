

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from ReplayMemory import *
from DQN import *
from process_data import *
from Vocabulary import *

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
TARGET_UPDATE = 10

VOCAB_SAMPLE = 20
VOCAB_CALCULATE = 1

DATA_FILE = '/mnt/c/files/research/projects/aud_neuro/data/WSJ_phones.txt'
PHONES = '/mnt/c/files/research/projects/aud_neuro/data/phones.txt'

to_print = False

def optimize_model():

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack(tuple([s for s in batch.next_state
                                                if s is not None]))
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
    loss = loss.double()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(0)[1].view(1, 1)
    else:
        action =  torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return action
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


n_actions = 2


# Get length of file to find number of episodes

input_data, phones2vectors, vectors2phones = initialize_data(DATA_FILE, PHONES)

num_inputs = len(phones2vectors.keys())
num_episodes = len(input_data)

policy_net = DQN(num_inputs, n_actions).to(device)
target_net = DQN(num_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

episode_durations = []

vocab = Vocabulary(VOCAB_SAMPLE, VOCAB_CALCULATE)

for i_episode in range(num_episodes):
    # Initialize the environment and state

    current_episode = get_episode(input_data, i_episode)
    state, symbol = get_state(current_episode, 0, phones2vectors)
    vocab.new_episode()

    if to_print:
        print('-----------------------------------------------------------')
        print('episode num: '+str(i_episode) + ', episode_length: '+str(len(current_episode)-1))
        print(current_episode)

    for t in count():
        # Select and perform an action
        action = select_action(state)

        reward = vocab.step(action, state)


        if to_print:
            if action == 0:
                print('state: '+symbol + ', action: continue')
            else:
                print('state: '+symbol + ', action: segment')
                print('word: ' + vocab.get_previous_word(vectors2phones) + ', reward: ' + str(reward))

        reward = torch.tensor([reward], device=device, dtype=torch.float64)

        done = t+1 == len(current_episode)-1
        # Observe new state
        if not done:
            next_state, symbol = get_state(current_episode, t+1, phones2vectors)
        else:
            next_state, symbol = None, None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 100 == 0:
        vocab_size, avg_size = vocab.get_info()

        print('episode: '+str(i_episode) +', vocab size: '+str(vocab_size)+ ', average word size: ' + str(avg_size))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        print()
print('Complete')
#env.render()
#env.close()
#plt.ioff()
#plt.show()
