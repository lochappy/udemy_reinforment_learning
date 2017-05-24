#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:14:07 2017

@author: lochappy
"""
import grid_world
import numpy as np

#discount factor
gamma = 0.9
EPS = 0.0001 

SET_OF_ACTIONS = set(['U','R','L','D'])

def random_action(a):
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        setA = list(SET_OF_ACTIONS)
        setA.remove(a)
        return np.random.choice(setA)

#playgame
def playGame(grid,policy):
    
    available_states = grid.actions.keys()
    #randomly select the start state
    s = available_states[np.random.choice(len(available_states))]
    grid.set_state(s)
    #print s
    
    rewards = [(s,0)]
    while(not grid.game_over()):
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        rewards.append((s,r))    
    
    returned_s_g = []    
    #reverse the sequence from end to start
    rewards.reverse()
    G = 0.#first G(t+1) is the reward of the terminal state, which is 0.0 by definition
    for s,r in rewards:
        returned_s_g.append((s,G))
        G = r + gamma*G # G(t) = r + gamma*G(t+1)
    returned_s_g.reverse()
    
    #cuz the first state doest not move at all G is not valid
    #returned_s_g = returned_s_g[:-1]
    
    return returned_s_g


############### Monte Carlo Policy Evaluation #####################

#create the standart grid world
#g = grid_world.negative_grid(step_cost=-0.1)
g = grid_world.standard_grid()

# state -> action
# found by policy_iteration_random on standard_grid
# MC method won't get exactly this, but should be close
# values:
# ---------------------------
#  0.43|  0.56|  0.72|  0.00|
# ---------------------------
#  0.33|  0.00|  0.21|  0.00|
# ---------------------------
#  0.25|  0.18|  0.11| -0.17|
# policy:
# ---------------------------
#   R  |   R  |   R  |      |
# ---------------------------
#   U  |      |   U  |      |
# ---------------------------
#   U  |   L  |   U  |   L  |
policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
  }

print "======== Policy ==========="
grid_world.print_policy(policy,g)

print "======== Rewards ==========="
grid_world.print_rewards(g)

#retrieve all the available states in g 
available_actions = g.actions

#retrieve all the available states in g 
available_states = g.all_states()

# randomly initialize the value of each state
# The values of terminal state must be 0
value_state = {s:0 for s in available_states}

print "======== Initial Value ==========="
grid_world.print_values(value_state,g)

all_G = {}
for s in available_states:
    all_G[s] = []
num_episo = 5000
for i in xrange(num_episo):
    
    seen_s = set()
    returned_G = playGame(g,policy)
    for s, G in returned_G:
        if s not in seen_s:
            seen_s.add(s)
            all_G[s].append(G)
            value_state[s] = np.mean(all_G[s])
              
print "======== Value ==========="
grid_world.print_values(value_state,g)







        
        
                
    
    