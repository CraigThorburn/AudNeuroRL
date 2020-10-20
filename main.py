# import math
# import random
# import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import sys
# from collections import namedtuple
from itertools import count

# import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
from ReplayMemory import *
from DQN import *
# from process_data import *
from Vocabulary import *
from Data import *
import time
from params import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-debug", help="run with debugging output on")
parser.add_argument("host", help="server on which code is run")
parser.add_argument("-overwrite", help="overwrite any existing output files")
args = parser.parse_args()

if args.host == 'local':
    ROOT = '/mnt/c/files/research/projects/aud_neuro/data/'
elif args.host == 'clip':
    ROOT = '/fs/clip-realspeech/projects/aud_neuro/models/dqn/WSJ/'

to_print = args.debug

DATA_PATH = ROOT + DATA_FILE
PHONE_PATH = ROOT + PHONE_FILE
VOCAB_PATH = ROOT + VOCAB_FILE


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    policy_net.train()

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack(tuple([s for s in batch.next_state
                                               if s is not None]))
    non_final_h0 = torch.stack(tuple([h[0] for h in batch.next_hidden
                                      if h is not None]))
    non_final_c0 = torch.stack(tuple([c[1] for c in batch.next_hidden
                                      if c is not None]))
    non_final_hidden = (non_final_h0, non_final_c0)

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    h0_batch = torch.cat(tuple([h[0] for h in batch.hidden]))
    c0_batch = torch.cat(tuple([h[1] for h in batch.hidden]))
    hidden_batch = (h0_batch, c0_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values, _ = policy_net(state_batch, hidden_batch)
    state_action_values = state_action_values.reshape(128, 2).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values_network, _ = target_net(non_final_next_states, non_final_hidden)

    next_state_values[non_final_mask] = next_state_values_network.max(-1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # TODO: Make sure loss is functioning correctly
    policy_net.train()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
    loss = loss.double()
    # Optimize the model
    optimizer.zero_grad()
    # print(policy_net.training)
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state, hidden):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        network_return, hidden = policy_net(state, hidden)
        network_return = network_return.max(-1)[1].view(1, 1)
    if sample > eps_threshold:
        action = network_return
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action, hidden


# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#    from IPython import display


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

n_actions = 2

# Get length of file to find number of episodes
print('loading data')
data = Data(DATA_PATH, PHONE_PATH)
data.load_data()
print('data loaded')
# input_data, phones2vectors, vectors2phones = initialize_data(DATA_PATH, PHONES)

num_inputs = data.num_inputs()  # len(phones2vectors.keys())
print('num inputs: ' + str(num_inputs))
num_episodes = len(data)  # len(input_data)
print('num episodes: ' + str(num_episodes))

policy_net = DQN(num_inputs, n_actions).to(device)
target_net = DQN(num_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# CHANGE: from RMSprop to SGD
optimizer = optim.SGD(policy_net.parameters(), lr=0.9)
memory = ReplayMemory(10000)
torch.backends.cudnn.enabled = False
# TODO: Check what exactly this is doing ^^^

policy_net.train()
print('model and memory initialized')

steps_done = 0
episode_durations = []
vocab = LoadedVocabulary(VOCAB_SAMPLE, VOCAB_CALCULATE, MEM_SIZE, TOKEN_TYPE)
vocab.load_vocabulary(VOCAB_PATH, data.get_phones2vectors())
# vocab = PhoneVocabulary(VOCAB_SAMPLE, VOCAB_CALCULATE, MEM_SIZE, TOKEN_TYPE)

if args.overwrite:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(ROOT + CORPUS + '_output.txt', write_method)
datafile = open(ROOT + CORPUS + '_data.txt', write_method)
outfile.close()
datafile.close()

to_output = []
to_data = []
print('running')

tic = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    h0 = torch.randn(1, 1, 112).to(device)
    c0 = torch.randn(1, 1, 112).to(device)
    hidden = (h0, c0)

    total_reward = 0

    # current_episode = get_episode(input_data, i_episode)
    episode_length = data.current_episode_length()
    state, symbol = data.get_state().to(device), data.get_symbol()  # current_episode, 0, phones2vectors)
    vocab.reset_local_memory()

    if to_print:
        print('-----------------------------------------------------------')
        print('episode num:  ' + str(i_episode) + ', episode_length:  ' + str(len(data.get_episode()) - 1))
        print(data.get_episode())

    for t in count():
        # Select and perform an action

        done = t + 1 == episode_length - 1

        action, next_hidden = select_action(state, hidden)
        if done:
            action = torch.tensor([[1]], device=device, dtype=torch.long)
        reward = vocab.step(action, state)
        total_reward += reward
        ### IMPORTANT: Reset hidden state if segment
        if action == 1:
            h0 = torch.randn(1, 1, 112).to(device)
            c0 = torch.randn(1, 1, 112).to(device)
            next_hidden = (h0, c0)

        if action == 0:
            pass
            # print('state:  ' + symbol + ', action: continue')
        else:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            if to_print:
                # print('state:  ' + symbol + ', action: segment')
                print('word: ' + vocab.get_word_string(vocab.get_previous_word(), data.get_vectors2phones()) + \
                      ', reward: ' + str(reward) + ', threshold: ' + str(eps_threshold))
            to_output.append('word: ' + vocab.get_word_string(vocab.get_previous_word(), data.get_vectors2phones()) + \
                             ', reward: ' + str(reward))

        reward = torch.tensor([reward], device=device, dtype=torch.float64)

        # Observe new state
        if not done:
            data.advance_state()
            next_state, symbol = state, symbol = data.get_state().to(device), data.get_symbol()
        else:
            h0 = torch.zeros(1, 1, 112).to(device)
            c0 = torch.zeros(1, 1, 112).to(device)
            next_hidden = (h0, c0)
            next_state, symbol, next_hidden = None, None, None

        # Store the transition in memory
        memory.push(state, action, next_state, reward, hidden, next_hidden)

        # Move to the next state
        state = next_state
        hidden = next_hidden

        # Perform one step of the optimization (on the target network)
        policy_net.train()
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            to_output.append('\n')
            to_data.append([t + 1, total_reward])

            total_reward = 0

            if i_episode + 1 != num_episodes:
                data.advance_episode()
            # plot_durations()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % UPDATES == 0:
        outfile = open(ROOT + CORPUS + '_output.txt', 'a+')
        outfile.write(''.join([i + '|' for i in to_output]))
        outfile.close()
        to_out = []

        datafile = open(ROOT + CORPUS + '_data.txt', 'a+')
        datafile.write(''.join([str(i[0]) + ' ' + str(i[1]) + '\n' for i in to_data]))
        datafile.close()
        to_data = []

        torch.save(policy_net.state_dict(), ROOT + '/checkpoints/' + CORPUS + 'model.pt')

        vocab_size, avg_size = vocab.get_info()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        toc = time.time()
        time_passed = toc - tic
        time_remaining = ((time_passed / (i_episode + 1)) * num_episodes - time_passed) / 60

        print(
            'episode: ' + str(i_episode) + ', vocab size: ' + str(vocab_size) + ', average word size: ' + str(avg_size) \
            + ', percent complete: ' + str(
                math.ceil((i_episode / num_episodes) * 100)) + \
            ', time remaining: ' + str(int(time_remaining)) + ' minutes')

print('model complete')
vocab_size, avg_size = vocab.get_info()
print(
    'total episodes: ' + str(i_episode) + ', vocab size: ' + str(vocab_size) + ', average word size: ' + str(avg_size))
print('saving vocab')
# vocab.save_vocab(VOCAB_PATH)
print('vocab saved')
print('done')
# env.render()
# env.close()
# plt.ioff()
# plt.show()

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Reward is not cumulative over single episode
# TODO: Memory is only cleared at end of episode
