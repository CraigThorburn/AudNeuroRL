import numpy
class Vocabulary(object):

    def __init__(self, sample_size, calculation_size):
        self.sample_size = sample_size
        self.calculation_size = calculation_size
        self.memory = []

    def push(self, word):
        """Saves a word."""
        self.memory.append(word)

    def get_reward(self,new_word):
        if len(self.memory)==0:
            return 0
        samples = random.sample(self.memory, self.calculation_size)
        rewards = []
        for s in samples:
            rewards.append(self.levenshteinDistanceDP(new_word, s))

        return(max(rewards))


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