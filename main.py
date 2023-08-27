# import libraries
import numpy as np
import matplotlib.pyplot as plt

# define the size of the field
field_rows = 10
field_cols = 10

# Q-values for each state and action pair: Q(s, a)
#   The third value is "action" dimension.
#   The value of each (state, action) pair is initialized to 0.
q_values = np.zeros((field_rows, field_cols, 5))

# actions: 0 = wait, 1 = up, 2 = right, 3 = down, 4 = left
actions = ["wait", "up", "right", "down", "left"]

# holds the rewards for each state
rewards = np.full((field_rows, field_cols), 20)


# function that determines if the robot has finished
def has_finished():  # TODO: cambiar esto
    return np.count_nonzero(rewards == 20) <= 0


def claim_reward(row, col, action):
    if has_finished():  # if the robot has finished, give it a big reward
        return 4000

    if action == 0:  # lazy penalty
        action_reward = -2
    else:  # reward for moving
        action_reward = -1  # move penalty
        action_reward += rewards[row, col]
        rewards[row, col] = 0
    return action_reward


# get a random starting location for the robot
def get_starting_location():
    # it can be any row
    initial_row = np.random.randint(field_rows)

    # start on the left or right side
    possible_inital_cols = [0, field_cols - 1]
    initial_col = possible_inital_cols[np.random.randint(len(possible_inital_cols))]

    return initial_row, initial_col


# epsilon greedy alorithm that will choose which action to take next
def get_next_action(current_row, current_col, epsilon):
    if np.random.random() < epsilon:  # choose best action
        return np.argmax(q_values[current_row, current_col])
    else:  # Random action
        return np.random.randint(len(actions))


# perform the action
def perform_action(current_row, current_col, action):
    new_row = current_row
    new_col = current_col
    action_reward = 0

    if actions[action] == "wait":
        pass
    elif actions[action] == "down":
        if current_row < field_rows - 1:
            new_row += 1
        else:
            action_reward = -1000
    elif actions[action] == "right":
        if current_col < field_cols - 1:
            new_col += 1
        else:
            action_reward = -1000
    elif actions[action] == "up":
        if current_row > 0:
            new_row -= 1
        else:
            action_reward = -1000
    elif actions[action] == "left":
        if current_col > 0:
            new_col -= 1
        else:
            action_reward = -1000

    return new_row, new_col, action_reward



def initialize_environment():
    initial_rewards = np.full((field_rows, field_cols), 20)
    return initial_rewards


# get the best path
def get_best_path(starting_row, starting_col):
    current_row, current_col = starting_row, starting_col

    best_path = []
    best_path.append([current_row, current_col])

    step = 0
    while not has_finished():
        step += 1
        # get the best action to take
        action = get_next_action(current_row, current_col, 1)
        current_row, current_col, reward = perform_action(current_row, current_col, action)
        rewards[current_row, current_col] = 0
        best_path.append([current_row, current_col])
        print(rewards)

    return best_path


# training parameters
EPSILON = 0.9  # the percentage of time when we should take the best action (instead of a random action)
DISCOUNT_FACTOR = 0.9  # discount factor for future rewards
LEARNING_RATE = 0.9  # the rate at which the AI agent should learn


# for plotting metrics
steps_per_episode = []


for episode in range(1000):
    rewards = initialize_environment()
    row, col = get_starting_location()
    rewards[row, col] = 0

    steps = 0

    while not has_finished():
        steps += 1
        action = get_next_action(row, col, EPSILON)

        old_row, old_col = row, col
        row, col, reward = perform_action(row, col, action)

        # receive the reward for moving the new state and calculate the temporal difference
        reward += claim_reward(row, col, action)
        old_q_value = q_values[old_row, old_col, action]
        temporal_difference = reward + (DISCOUNT_FACTOR * np.max(q_values[row, col])) - old_q_value

        # update the q-value for the previous state and action pair
        new_q_value = old_q_value + (LEARNING_RATE * temporal_difference)
        q_values[old_row, old_col, action] = new_q_value

    steps_per_episode.append(steps)
    print("Training episode: {}".format(episode) + " - Steps: {}".format(steps))

print("training complete!")


# Crear la gráfica
plt.plot(range(1000), steps_per_episode)
plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.title("Number of Steps per Episode")
plt.show()



#display a few shortest paths
rewards = initialize_environment()
rewards[0, 0] = 0
path = get_best_path(0,0)
print(path)

