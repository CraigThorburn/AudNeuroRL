#TODO: subsume these functions within Vocabulary

import numpy as np
import torch
def initialize_data(DATA_FILE, PHONES):
    with open(DATA_FILE, 'r') as f:
        input_data = f.read().splitlines()

    with open(PHONES, 'r') as f:
        phone_list = f.read().splitlines()

    num_phones = len(phone_list)

    phones2vectors = {}
    vectors2phones={}
    for i in range(num_phones):
        vector = np.zeros(num_phones)
        vector[i] += 1
        phones2vectors[phone_list[i]] = vector
        vectors2phones[str(vector)] = phone_list[i]

    return input_data, phones2vectors, vectors2phones

def get_episode(input_data, i_episode):
    current_episode = [i for i in input_data[i_episode].split(' ') if i != '']
    return current_episode

def get_state(current_episode, t, phone_vectors):
    current_state = phone_vectors[current_episode[t+1]]
    return torch.from_numpy(current_state).float(), current_episode[t+1]

