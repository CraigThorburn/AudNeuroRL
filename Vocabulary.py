import numpy
import random
import math
import torch

class Vocabulary(object):

    def __init__(self, sample_size, calculation_size, mem_limit, token_type):

        self.sample_size = sample_size
        self.calculation_size = calculation_size
        self.current_word = []
        self.memory = []
        self.previous_word = []
        self.mem_limit = mem_limit
        self.token_type = token_type
        if token_type == 'token':
            self.push = self.push_token
        elif token_type == 'item':
            self.push = self.push_token

    def __len__(self):
        return len(self.memory)

    def push_token(self, word):
        """Saves a word."""
        self.memory.append(word)
        if len(self.memory) > self.mem_limit:
            self.memory.pop(0)

    def push_type(self, word):
        """Saves a word."""
        if word not in self.memory:
            self.memory.append(word)
        if len(self.memory) > self.mem_limit:
            self.memory.pop(0)

    # def push(self, word):
    #    """Saves a word."""
    #    self.memory.append(word)
    #   if len(self.memory) > self.mem_limit:
    #      self.memory.pop(0)

    def step(self, action, state):
        self.current_word.append(state)
        if action == 0:
            reward = 0
        elif action == 1:
            reward = self.get_reward(self.current_word)
            self.push(self.current_word)
            self.previous_word = self.current_word
            self.current_word = []
        return reward

    def reset_local_memory(self):
        self.current_word = []
        self.previous_word = []

    def get_reward(self, new_word):
        if len(self.memory) == 0:
            return 0.0
        if self.sample_size > len(self.memory):
            this_sample = len(self.memory)
        else:
            this_sample = self.sample_size
        samples = random.sample(self.memory, this_sample)
        dists = []
        for s in samples:
            dists.append(self.distance(new_word, s))
            length = len(new_word)
            min_dist = min(dists)
            if min_dist == 0:
                reward = length# 10 * math.exp(0.5*length)
            else:
                reward = 0 #(2 / min_dist) * math.exp(0.5*length)
        return reward



    def get_info(self):
        vocab_size = self.__len__()
        total_size = 0.0
        for w in self.memory:
            total_size += len(w)
        avg_size = total_size / vocab_size
        return vocab_size, avg_size

    def get_current_word(self):
        return self.current_word

    def get_previous_word(self):
        return self.previous_word



class PhoneVocabulary(Vocabulary):

    def __init__(self, sample_size, calculation_size, mem_limit, token_type):
        super().__init__(sample_size, calculation_size, mem_limit, token_type)
        self.distance = self.levenshteinDistanceDP

    def calculate_unique_words(self, vectors2phones):
        #TODO: Add optimization for this
        self.word_dictionary = {}
        for word in self.memory:
            string = self.get_word_string(word, vectors2phones)
            if string not in self.word_dictionary.keys():
                self.word_dictionary[string] = 0
            self.word_dictionary[string] += 1
        self.word_dictionary = sorted(self.word_dictionary)

    def get_total_unique_words(self):
        return len(self.word_dictionary.keys())


    def save_vocab(self, filename):
        with open(filename, 'w+') as f:
            f.write(''.join([u + str(self.word_dictionary[u]) + '\n' for u in self.word_dictionary.keys()]))
            # TODO: Save words in order

    def get_word_string(self, word_vector, vectors2phones):
        w = ''.join([vectors2phones[str(s.cpu().data.numpy())] + ' ' for s in word_vector])
        return w

    def printDistances(self, distances, token1Length, token2Length):
        for t1 in range(token1Length + 1):
            for t2 in range(token2Length + 1):
                print(int(distances[t1][t2]), end=" ")
            print()

    def levenshteinDistanceDP(self, token1, token2):
        distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1

        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2

        a = 0
        b = 0
        c = 0

        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                diff = token1[t1 - 1] == token2[t2 - 1]
                if diff.all():
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        # self.printDistances(distances, len(token1), len(token2))
        return distances[len(token1)][len(token2)]

    def vectorDistance(self):
        pass


class VectorVocabulary(Vocabulary):

    def __init__(self, sample_size, calculation_size, mem_limit, token_type):
        super().__init__(sample_size, calculation_size, mem_limit, token_type)
        self.distance = self.vectorDistance



class LoadedVocabulary(PhoneVocabulary):
    def __init__(self, sample_size, calculation_size, mem_limit, token_type):
        super().__init__(sample_size, calculation_size, mem_limit, token_type)

    def load_vocabulary(self, vocab_file, phones2vectors):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(vocab_file, 'r') as f:
            all_vocab = f.read().splitlines()
            self.memory = [[torch.tensor(phones2vectors[p]).to(device) for p in w.split(' ')] for w in all_vocab]


    def step(self, action, state):
        self.current_word.append(state)
        if action == 0:
            reward = 0
        elif action == 1:
            reward = self.get_reward(self.current_word)
            self.previous_word = self.current_word
            self.current_word = []
        return reward

# TODO: Fix variable names around unique words
