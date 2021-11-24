import numpy as np
from Guy.Assignment_3.constructe_grid import grid_world, Rewards
num_of_states = 12

def value_iteration(threshold, gamma, use_random_state):
    V = np.zeros(num_of_states)
    V[4] = -1000
    V[9] = 1
    V[10] = -1

    if use_random_state:
        states_order = np.random.permutation(num_of_states)
    else:
        states_order = np.arange(num_of_states)

    policy = np.zeros(num_of_states)

    delta = threshold + 1
    step_counter = 0

    while threshold < delta:
        delta = 0
        for i in range(num_of_states):
            s = states_order[i]
            if s == 4 or s == 9 or s == 10:
                continue

            v = V[s]
            V[s] = np.max(np.dot(np.transpose(grid_world[s, :, :]), (Rewards + gamma*V)))
            delta = np.max([delta, np.abs(v - V[s])])

        step_counter = step_counter + 1

    for s in range(num_of_states):
        if s == 4 or s == 9 or s == 10:
            continue
        policy[s] = np.argmax(np.dot(np.transpose(grid_world[s, :, :]), (Rewards + gamma*V)))
    print(policy)

if __name__ == '__main__':
    value_iteration(1e-4, 1, 0)

