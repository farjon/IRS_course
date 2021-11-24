import numpy as np

np.random.seed(10)
grid_rows = 3
grid_cols = 4
number_of_S = 12
number_of_A = 4 # N = 0, E = 1, S = 2, W = 3
start_state = 3
obstacle_state = 5
termination_pos = 10
termination_neg = 11

Rewards = np.ones(number_of_S) * -0.04
Rewards[obstacle_state-1] = 0
Rewards[termination_pos-1] = 1
Rewards[termination_neg-1] = -1

p_act = np.array([0.8, 0.2, 0, 0])

help_mat_grid = np.zeros([grid_rows+2, grid_cols+2])
index = 0
for i in range(1, grid_cols+1):
    for j in range(1, grid_rows+1):
        index = index + 1
        if index == obstacle_state:
            continue
        help_mat_grid[j,i] = index

def check_cell(ind, cell, prob, prob_state_vector):
    if cell == 0:
        prob_state_vector[ind-1] = prob_state_vector[ind-1] + prob
    else:
        prob_state_vector[cell-1] = prob_state_vector[cell-1] + prob
    return prob_state_vector


def check_cells_probs(index, front_cell, right_cell, back_cell, left_cell):
    prob_state_vector = np.zeros(number_of_S)
    prob_state_vector = check_cell(index, front_cell, p_act[0], prob_state_vector)
    prob_state_vector = check_cell(index, right_cell, p_act[1], prob_state_vector)
    prob_state_vector = check_cell(index, back_cell, p_act[2], prob_state_vector)
    prob_state_vector = check_cell(index, left_cell, p_act[3], prob_state_vector)
    return prob_state_vector


grid_world = np.zeros([number_of_S, number_of_S, number_of_A])
index = 0
for a in range(number_of_A):
    for col in range(1, grid_cols + 1):
        for row in range(1, grid_rows + 1):
            index = index + 1
            if obstacle_state == index or termination_neg == index or termination_pos == index:
                continue
            n_cell = int(help_mat_grid[row-1,col])
            e_cell = int(help_mat_grid[row,col + 1])
            s_cell = int(help_mat_grid[row + 1,col])
            w_cell = int(help_mat_grid[row,col - 1])
            if a == 0:
                prob_state_vector = check_cells_probs(index, n_cell, e_cell, s_cell, w_cell)
            elif a == 1:
                prob_state_vector = check_cells_probs(index, e_cell, s_cell, w_cell, n_cell)
            elif a == 2:
                prob_state_vector = check_cells_probs(index, s_cell, w_cell, n_cell, e_cell)
            elif a == 3:
                prob_state_vector = check_cells_probs(index, w_cell, n_cell, e_cell, s_cell)
            grid_world[index-1, :,a] = prob_state_vector
    index = 0

def step(state, action):
    done = False
    next_state_dist = grid_world[state, :, action]
    next_state = np.random.choice(number_of_S, 1, True, next_state_dist).item()
    reward = Rewards[next_state]
    if next_state+1 == termination_pos or next_state+1 == termination_neg:
        done = True
    return next_state, reward, done

def generate_episode(policy):
    '''

    :param policy: a S x A matrix - the probability to choose action for each State
    :return:
    '''
    stop = 0
    current_state = start_state - 1
    episode = {
        'state': [],
        'action': [],
        'reward': []
    }
    episode['state'].append(current_state)
    while True:
        action = np.random.choice(number_of_A, 1, True, policy[current_state, :]).item()
        episode['action'].append(action)
        next_state, reward, stop = step(current_state, action)
        episode['reward'].append(reward)

        current_state = next_state
        episode['state'].append(next_state)
        if stop == 1:
            break

    return episode

def calculate_gain(episode, discount_factor, Q, learning_rate):
    '''

    :param episode:
    :param discount_factor:
    :return:
    '''
    visits = np.zeros([number_of_S, number_of_A])

    for t in range(len(episode['state']) - 1):
        state_t = episode['state'][t]
        action_t = episode['action'][t]
        gain = 0
        visits[state_t, action_t] += 1

        for k in range(t, len(episode['state']) - 1):
            reward_k = episode['reward'][k]
            gain = gain + (discount_factor ** (k - t)) * reward_k
        Q[state_t, action_t] = Q[state_t, action_t] + learning_rate * (gain - Q[state_t, action_t])
    return Q

def epsilon_greedy(Q, epsilon):
    policy = np.zeros([number_of_S, number_of_A])
    for state in range(Q.shape[0]):
        if state+1 == obstacle_state or state+1 == termination_neg or state+1 == termination_pos:
            continue
        else:
            action_star = int(np.max(Q[state,:]))
        for act in range(Q.shape[1]):
            if act == action_star:
                policy[state, act] = 1 - epsilon + (epsilon / Q.shape[1])
            else:
                policy[state, act] = epsilon / Q.shape[1]
    return policy

def monte_carlo_FV_GLIE(num_episodes, discount_factor, learning_rate):
    Q = np.zeros([number_of_S, number_of_A])
    for i in range(num_episodes):
        epsilon = 1 / np.sqrt(i+1)/10 + 1
        policy = epsilon_greedy(Q, epsilon)
        episode = generate_episode(policy)
        Q = calculate_gain(episode, discount_factor, Q, learning_rate)
        print(Q)
    policy_star = np.argmax(Q, axis=1)
    # value_star = np.max(Q, axis=1)
    print(policy_star)
    # print(value_star)

def main():
    num_episodes = 30000
    discount_factor = 0.9
    learning_rate = 0.0015
    monte_carlo_FV_GLIE(num_episodes, discount_factor, learning_rate)

if __name__ == '__main__':
    main()


