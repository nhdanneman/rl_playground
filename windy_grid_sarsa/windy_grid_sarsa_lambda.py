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



### Tabular SARSA-lambda
# For every state, for every action, we have estimates of value
# Policy (e-greedy): do the highest-valued policy, except e fraction of the time choose at random
# Undiscounted episodic task: constant -1 reward until goal

# in the lambda version, we track an activity trace every episode
# all updates accrue to discounted activity trace weights that top out at 1

# policy is ripped right out of the q-table, which is 2-level dict of state:action:estimate
# (x,y):(delta_x, delta_y):float

possible_actions = [(-1, 0), (0, 1), (1,0), (0, -1)]
# e: fraction of the time to make a random action (instead of greedy)
e = 0.1
# alpha: discount for rewards
alpha = 0.1
# LAMBDA: how many steps back in time the activity trace lasts
LAMBDA = 4
# gamma: how much the weight of new rewards accrues each step back through the trace
gamma = .6


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

# similar structure as q_table, dict of state:action:[value, steps_back]
def update_trace(trace, LAMBDA, gamma, curr_state, curr_action):
    # for every state, for every action...
    for s in list(trace.keys()):
        for a in list(trace[s].keys()):
            # if steps_back + 1 > LAMBDA, delete entry, and consider deleting entry one level up
            if trace[s][a][1] + 1 > LAMBDA:
                del trace[s][a]
                if len(list(trace[s].keys())) == 0:
                    del trace[s]
            # else, increment steps_back and update value
            else:
                trace[s][a][0] = trace[s][a][0]*gamma
                trace[s][a][1] += 1
    # add the current state-action, with value=1 and steps_back=0
    # this formulation ensures trace values are never > 1
    if curr_state in trace:
        trace[curr_state][curr_action] = [1, 0]
    else:
        trace[curr_state] = dict()
        trace[curr_state][curr_action] = [1, 0]
    return trace


def episode(e, alpha, LAMBDA, gamma, possible_actions):
    n_steps = 0
    # initialize current state (start) and action(random)
    current_action = possible_actions[np.random.choice(range(len(possible_actions)))]
    current_state = (0, 3)
    # initialize a dict to store activity traces
    # similar structure as q_table, dict of state:action:(value, steps_back)
    trace = dict()

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
        # SARSA-lambda Update
        # First, update the eligibility trace
        trace = update_trace(trace, LAMBDA, gamma, current_state, current_action)
        if n_steps < 6:
            print(trace)
        # Next, compute the full update
        val_curr_sa = q_table[current_state][current_action]
        val_next_sa = q_table[next_state][next_action]
        full_update = val_curr_sa + alpha * (-1 + val_next_sa - val_curr_sa)
        # now, update the q_table for all state-actions according to their values in the eligibility trace
        for s in list(trace.keys()):
            for a in list(trace[s].keys()):
                q_table[s][a] = full_update * trace[s][a][0]
        ### ORIG: q_table[current_state][current_action] = update
        # wrapping up the loop
        if next_state == (7,3):
            goal=True
        current_state = next_state
        current_action = next_action
    return n_steps


# try one episode:
ns = episode(e, alpha, LAMBDA, gamma, possible_actions)

q_table = mk_new_q_table(possible_actions)
# Ok, now try running until convergence:
for i in range(1000):
    ns = episode(e, alpha, LAMBDA, gamma, possible_actions)
    if i % 1 == 0:
        print(i)
        print(ns)
        if ns < 16:
            break



