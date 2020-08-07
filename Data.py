import torch
import numpy as np

class Data(object):

    def __init__(self, DATA_FILE, PHONE_FILE):
        self.data_file = DATA_FILE
        self.phone_file = PHONE_FILE

        self.previous_word = []
        self.phones2vectors = {}
        self.vectors2phones = {}
        self.all_episodes = []

        self.current_episode = None
        self.current_state = None
        self.current_symbol = None

        self.episode_num = 0
        self.state_num = 1

    def load_data(self):
        with open(self.data_file, 'r') as f:
            input_data = f.read().splitlines()

        with open(self.phone_file, 'r') as f:
            phone_list = f.read().splitlines()

        num_phones = len(phone_list)


        for i in range(num_phones):
            vector = np.zeros(num_phones)
            vector[i] += 1
            self.phones2vectors[phone_list[i]] = vector
            self.vectors2phones[str(vector)] = phone_list[i]

        self.all_episodes = input_data

        self.current_episode = [i for i in self.all_episodes[self.episode_num].split(' ') if i != '']
        self.set_first_state()

    def advance_episode(self):
        self.episode_num += 1
        self.current_episode = [i for i in self.all_episodes[self.episode_num].split(' ') if i != '']
        self.set_first_state()

    def set_first_state(self):
        self.state_num=1
        self.current_symbol = self.current_episode[self.state_num]
        self.current_state = torch.from_numpy(self.phones2vectors[self.current_episode[self.state_num]]).float()

    def get_episode(self):
        return self.current_episode

    def current_episode_length(self):
        return len(self.current_episode)

    def advance_state(self):
        self.state_num+=1
        self.current_symbol = self.current_episode[self.state_num]
        self.current_state = torch.from_numpy(self.phones2vectors[self.current_episode[self.state_num]]).float()

    def get_state(self):
        return self.current_state

    def get_symbol(self):
        return self.current_symbol

    def get_vectors2phones(self):
        return self.vectors2phones

    def get_phones2vectors(self):
        return self.phones2vectors

    def __len__(self):
        return len(self.all_episodes)

    def num_inputs(self):
        return len(self.phones2vectors.keys())
