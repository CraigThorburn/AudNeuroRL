def initialize_data(DATA_FILE, PHONES):
    with open(DATA_FILE, 'r') as f:
        input_data = f.read().splitlines()

    with open(PHONES, 'r') as f:
        phone_list = f.read().splitlines()

    num_phones = len(phone_list)

    phone_vectors = {}
    for i in range(num_phones):
        vector = np.zeros(num_phones)
        vector[i+1] += 1
        phone_vectors[phone_list[i]] = vector

    return input_data, phone_vectors
def get_episode():
    current_episode = input_data[i_episode]
    return current_episode

def get_state():
    current_state = input_data[t]
    return current_state

