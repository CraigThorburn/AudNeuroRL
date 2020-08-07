import numpy
import random
import math
class Vocabulary(object):

    def __init__(self, sample_size, calculation_size):

        self.sample_size = sample_size
        self.calculation_size = calculation_size
        self.word_vector = []
        self.memory = []
        self.unique_words = {}
        self.previous_word = []


    def push(self, word):
        """Saves a word."""
        self.memory.append(word)

    def push_to_unique_words(self, word_string):
        if word_string not in self.unique_words.keys():
            self.unique_words[word_string]=1
        else:
            self.unique_words[word_string] +=1


    def reset_local_memory(self):
        self.word_vector = []
        self.previous_word = []

    def get_reward(self,new_word):
        if len(self.memory)==0:
            return 0.0
        if self.sample_size > len(self.memory):
            this_sample = len(self.memory)
        else:
            this_sample = self.sample_size
        samples = random.sample(self.memory, this_sample)
        dists = []
        for s in samples:
            dists.append(self.levenshteinDistanceDP(new_word, s))
            length = len(new_word)
            min_dist = min(dists)
            if min_dist==0:
                reward = 2*math.exp(length)
            else:
                reward = (1/min_dist)*math.exp(length)
        return reward


    def __len__(self):
        return len(self.memory)

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

        #self.printDistances(distances, len(token1), len(token2))
        return distances[len(token1)][len(token2)]

    def printDistances(self, distances, token1Length, token2Length):
        for t1 in range(token1Length + 1):
            for t2 in range(token2Length + 1):
                print(int(distances[t1][t2]), end=" ")
            print()

    def step(self, action, state):
        self.word_vector.append(state)
        if action == 0:
            reward = 0
        elif action == 1:
            reward = self.get_reward(self.word_vector)
            self.push(self.word_vector)
            self.previous_word = self.word_vector
            self.word_vector = []
        return reward

    def get_previous_word(self, vectors2phones):
        w = ''.join([vectors2phones[str(s.data.numpy())] + ' ' for s in self.previous_word])
        return w

    def get_info(self):
        vocab_size = self.__len__()
        total_size = 0.0
        for w in self.memory:
            total_size+=len(w)
        avg_size = total_size/vocab_size
        unique_words = len(self.unique_words.keys())
        return vocab_size, avg_size, unique_words

    def save_vocab(self, filename):
        with open(filename, 'w+') as f:
            f.write(''.join([u + str(self.unique_words[u])+ '\n' for u in self.unique_words.keys()]))
            # TODO: Save words in order

# TODO: Fix variable names around unique words