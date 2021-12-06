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

# create a reward matrix
Rewards = np.ones(number_of_S) * -0.04
Rewards[obstacle_state-1] = 0
Rewards[termination_pos-1] = 1
Rewards[termination_neg-1] = -1

# noisy-action distribution
p_act = np.array([0.8, 0.2, 0, 0])

# using an help matrix to contain the world
help_mat_grid = np.zeros([grid_rows+2, grid_cols+2])
index = 0
for i in range(1, grid_cols+1):
    for j in range(1, grid_rows+1):
        index = index + 1
        if index == obstacle_state:
            continue
        help_mat_grid[j,i] = index

def check_cell(cell, next_cell, prob, prob_state_vector):
    '''
    check if the cell is within the grid_world. If the next state is outside of the grid,
    the robot will bump a wall, and remains in the same cell.
    :param cell: an int indicating the current state (within 0-S)
    :param next_cell: an int indicating the next state
    :param prob: the probability to go in a specific direction
    :param prob_state_vector: a vector that store the probabilities for each state and next state
    :return:
    '''
    if next_cell == 0:
        prob_state_vector[cell-1] = prob_state_vector[cell-1] + prob
    else:
        prob_state_vector[next_cell-1] = prob_state_vector[next_cell-1] + prob
    return prob_state_vector


def check_cells_probs(index, front_cell, right_cell, back_cell, left_cell):
    '''
    update the current state probabilities for next state, given a specific chosen action
    :param index: current state
    :param front_cell: as the name indicates - the front cell
    :param right_cell: as the name indicates - the right cell
    :param back_cell: as the name indicates - the back cell
    :param left_cell: as the name indicates - the left cell
    :return:
    '''
    prob_state_vector = np.zeros(number_of_S)
    prob_state_vector = check_cell(index, front_cell, p_act[0], prob_state_vector)
    prob_state_vector = check_cell(index, right_cell, p_act[1], prob_state_vector)
    prob_state_vector = check_cell(index, back_cell, p_act[2], prob_state_vector)
    prob_state_vector = check_cell(index, left_cell, p_act[3], prob_state_vector)
    return prob_state_vector


# create the grid world
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
