'''SARSA RL Algo to solve Windy Gridworld
Sutton and Bartow pp. 146-147'''


import numpy as np
import matplotlib.pyplot as plt


# States are tuples (x,y)
# Actions are tuples (delta_x, delta_y)

### Environment
# takes a state and action, returns a state
# input [(state), (action)]   e.g. [(2,2),(1,0)]
# output (new state)   e.g (3,2)
def environment(state, action):
    # y 0-6
    # x 0-9
    # 'wind' is a function of beginning state
    # x {3, 4, 5, 8} add 1 to row
    # x {6,7} add 2 to row
    # start is (0,3)
    # goal is (7,3)

    # Check state. If wind, apply wind. If you're out-of-bounds, move that index in-bounds.
    one_wind = {3,4,5,8}
    two_wind = {6,7}

    x, y = state

    if x in one_wind:
        y += 1
    if x in two_wind:
        y += 2

    x += action[0]
    y += action[1]

    if x < 0:
        x = 0
    if x > 9:
        x = 9
    if y < 0:
        y=0
    if y > 6:
        y = 6

    out = (x,y)
    return(out)

'''
# quick checks
# easy cases
environment((1,3), (1,0))
environment((1,3), (0,1))

# non-wind bounds
environment((0,0), (-1,0))
environment((0,1), (0,-1))

# wind
environment((3,1), (1,0))
environment((6,1), (1,0))
'''

### Tabular SARSA
# For every state, for every action, we have estimates of value
# Policy (e-greedy): do the highest-valued policy, except e fraction of the time choose at random
# Undiscounted episodic task: constant -1 reward until goal

# policy is ripped right out of the q-table, which is 2-level dict of state:action:estimate
# (x,y):(delta_x, delta_y):float

possible_actions = [(-1, 0), (0, 1), (1,0), (0, -1)]
e = 0.1
alpha = 0.1


def mk_new_q_table(possible_actions):
    q_table = dict()
    rx = range(10)
    ry = range(7)
    for x in rx:
        for y in ry:
            if (x,y) not in q_table:
                q_table[(x,y)] = dict()
            for ac in possible_actions:
                q_table[(x,y)][ac] = 0
    return q_table


q_table = mk_new_q_table(possible_actions)

def episode(e, alpha, possible_actions):
    n_steps = 0
    # initialize current state (start) and action(random)
    current_action = possible_actions[np.random.choice(range(len(possible_actions)))]
    current_state = (0, 3)

    # Each episode you start, do updates, until goal.
    goal = False
    while goal != True:
        n_steps += 1
        # The next state is a function of current state and action
        next_state = environment(current_state, current_action)
        # To choose the next action via e-greedy strategy, first, decide whether max or random
        take_random_action = np.random.binomial(1,e)==1
        # random action condition
        if take_random_action:
            next_action = possible_actions[np.random.choice(range(len(possible_actions)))]
        # maximal action
        else:
            # fetch values from q-table
            vals = list(q_table[next_state].values())
            maximal_vals = np.where(vals==np.max(vals))[0]
            choice = np.random.choice(maximal_vals)
            next_action = possible_actions[choice]
        # SARSA Update
        val_curr_sa = q_table[current_state][current_action]
        val_next_sa = q_table[next_state][next_action]
        update = val_curr_sa + alpha * (-1 + val_next_sa - val_curr_sa)
        q_table[current_state][current_action] = update
        # wrapping up the loop
        if next_state == (7,3):
            goal=True
        current_state = next_state
        current_action = next_action
    return n_steps


# How many iterations does it take to converge (with bounds)
iterations = 20
mat = np.zeros((iterations, 100))

for i in range(iterations):
    print(i)
    q_table = mk_new_q_table(possible_actions)
    for j in range(999):
        if j % 10 == 0:
            ns = episode(0, alpha, possible_actions)
            ind = int(j/10)
            mat[i,ind] = ns
        else:
            ns = episode(e, alpha, possible_actions)


means = np.apply_along_axis(np.mean, 0, mat)
iters = range(0, 1000, 10)
lower = np.apply_along_axis(lambda x: np.quantile(x, 0.06), 0, mat)
upper = np.apply_along_axis(lambda x: np.quantile(x, 0.94), 0, mat)

plt.scatter(iters, means, color='k')
plt.yscale('log')
for i in range(len(iters)):
    plt.plot([iters[i], iters[i]], [lower[i], upper[i]], color='k')
plt.show()



