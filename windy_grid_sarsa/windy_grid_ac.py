'''Actor-Critic Algo to solve Windy Gridworld
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


### Actor-Critic Method
# Idea: Actor makes actions based on preference scores
# Critic evaluates deltas between V(current state) and V(next state)
# That diff is used to update the preference scores

possible_actions = [(-1, 0), (0, 1), (1,0), (0, -1)]


def mk_new_v_table():
    v_table = dict()
    rx = range(10)
    ry = range(7)
    for x in rx:
        for y in ry:
            if (x,y) not in v_table:
                v_table[(x,y)] = 0
    return v_table


def mk_new_actor_table(possible_actions):
    actor_table = dict()
    rx = range(10)
    ry = range(7)
    for x in rx:
        for y in ry:
            if (x,y) not in actor_table:
                actor_table[(x,y)] = dict()
            for ac in possible_actions:
                actor_table[(x,y)][ac] = 0
    return actor_table

def choose_action(actor_table, current_state):
    vals = list(actor_table[current_state].values())
    prs = []
    for i in range(len(vals)):
        pr_i = np.exp(vals[i]) / np.sum([np.exp(vals[x]) for x in range(len(vals)) if x != i])
        prs.append(pr_i)
    prs = prs / np.sum(prs)
    choice_ind = np.random.choice(list(range(len(possible_actions))), size=1, replace=False, p=prs)[0]
    action = possible_actions[choice_ind]
    return action


actor_table = mk_new_actor_table(possible_actions)
v_table = mk_new_v_table()

def episode(possible_actions, beta):
    n_steps = 0
    # initialize current state (start) and action(random)
    current_state = (0, 3)
    # Each episode you start, do updates, until goal.
    goal = False
    while goal != True:
        n_steps += 1
        action = choose_action(actor_table, current_state)
        next_state = environment(current_state, action)
        TD_error = -1 + v_table[next_state] - v_table[current_state]
        actor_table[current_state][action] += beta*TD_error

        # the v-table needs an update...um...
        update = v_table[current_state] + .1 * (-1 + v_table[next_state] - v_table[current_state])
        v_table[current_state] = update

        if next_state == (7,3):
            goal=True
        current_state = next_state
    return n_steps


# episode(possible_actions, .1)

for i in range(1000):
    ns = episode(possible_actions, .1)
    if i % 10 == 0:
        print(i)
        print(ns)


