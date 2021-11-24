import numpy as np
from Guy.Assignment_3.constructe_grid import grid_world, Rewards
np.random.seed(10)
# ---------------- grid world attributes ----------------
grid_rows = 3
grid_cols = 4
number_of_S = 12
number_of_A = 4 # N = 0, E = 1, S = 2, W = 3
start_state = 3
obstacle_state = 5
termination_pos = 10
termination_neg = 11


def step(state, action):
    '''
    a specific step function within an episode
    :param state: current state
    :param action:
    :return:
    '''
    done = False
    next_state_dist = grid_world[state, :, action]
    next_state = np.random.choice(number_of_S, 1, True, next_state_dist).item()
    reward = Rewards[next_state]
    if next_state+1 == termination_pos or next_state+1 == termination_neg:
        done = True
    return next_state, reward, done

def generate_episode(policy):
    '''
    an episode generation function
    :param policy: a S x A matrix - the probability to choose action for each state
    :return: a complete episode from state state to termination
    '''
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

def calculate_gain(episode, discount_factor, Q, visits, learning_rate):
    '''
    calculate the gain of an episode, use either the visits matrix or the learning rate to to update the Q value
    :param episode: a complete episode of state, actions, and rewards
    :param discount_factor: how to address to future rewards
    :param Q: Q-values for state-action pairs
    :param visits: a matrix for storing the visits of state-action pairs
    :param learning_rate:
    :return:
    '''
    for t in range(len(episode['state']) - 1):
        state_t = episode['state'][t]
        action_t = episode['action'][t]
        gain = 0
        visits[state_t, action_t] += 1

        for k in range(t, len(episode['state']) - 1):
            reward_k = episode['reward'][k]
            gain = gain + (discount_factor ** (k - t)) * reward_k
        Q[state_t, action_t] = Q[state_t, action_t] + (1/visits[state_t, action_t]) * (gain - Q[state_t, action_t])
    return Q, visits

def epsilon_greedy(Q, epsilon):
    '''

    :param Q:
    :param epsilon:
    :return:
    '''
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
    '''
    Solving a monte-carlo first visit GLIE algorithm
    :param num_episodes: number of episode to learn from
    :param discount_factor: the discount factor for future rewards during each episode
    :param learning_rate: the learning rate between episodes
    :return:
    '''
    visits = np.zeros([number_of_S, number_of_A])
    Q = np.zeros([number_of_S, number_of_A])
    for i in range(num_episodes):
        epsilon = 1 / np.sqrt(i+1)/10 + 1
        policy = epsilon_greedy(Q, epsilon)
        episode = generate_episode(policy)
        print(episode['state'])

        Q, visits = calculate_gain(episode, discount_factor, Q, visits, learning_rate)
    policy_star = np.argmax(Q, axis=1)
    print(policy_star)

def main():
    num_episodes = 30000
    discount_factor = 0.9
    learning_rate = 0.0015
    monte_carlo_FV_GLIE(num_episodes, discount_factor, learning_rate)

if __name__ == '__main__':
    main()


