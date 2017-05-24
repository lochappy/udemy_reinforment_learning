#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:14:07 2017

@author: lochappy
"""
import grid_world
import numpy as np

#create the standart grid world
g = grid_world.standard_grid()

#retrieve all the available states in g 
available_states = g.all_states()
print available_states

#discount factor
gamma = 0.9
EPS = 0.001 

#retrieve all the available states in g 
available_actions = g.actions
print available_actions

state_transition_prob = 1 # p(s',r|s,a) = 1 this is the determinstic case
                             
# randomly initialize the value of each state
# The values of terminal state must be 0
value_state = {s:np.random.randn() for s in available_states}
for s in available_states:
    if s not in available_actions: # this is the terminal state
      value_state[s] = 0
               
### fixed policy ###
policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
}
grid_world.print_policy(policy, g)


iteration = 0
while(1):
    print('---- Iteration {} -----'.format(iteration))
    maxdiff = 0.0
    for s in available_states:
        if s in policy:
            
            v_old = value_state[s]
            p_a = 1.

            g.set_state(s)
            reward = g.move(policy[s])
            new_s = g.current_state()
            value_state[s] = p_a*state_transition_prob*(reward + gamma*value_state[new_s])
            
            maxdiff = max(maxdiff,abs(value_state[s] - v_old))

    grid_world.print_values(value_state,g)
    print('---- Maxdiff {} = {} -----'.format(iteration,maxdiff))
    iteration += 1
    
    if maxdiff < EPS:
        break
        
        
                
    
    