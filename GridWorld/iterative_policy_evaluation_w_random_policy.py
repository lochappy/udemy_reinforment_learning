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
gamma = 1.0
EPS = 0.0001 

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
      
grid_world.print_values(value_state,g)
               
#### random uniform policy ####
iteration = 0
while(1):
    print('---- Iteration {} -----'.format(iteration))
    maxdiff = 0.0
    for s in available_states:
        if s in available_actions:
            
            v_old = value_state[s]
            p_a = 1./ len(available_actions[s])
            
            v_new = 0.0
            for action in available_actions[s]:
                g.set_state(s)
                reward = g.move(action)
                new_s = g.current_state()
                
                v_new += p_a*state_transition_prob*(reward + gamma*value_state[new_s])
            value_state[s] = v_new 
            
            maxdiff = max(maxdiff,abs(v_new - v_old))

    grid_world.print_values(value_state,g)
    print('---- Maxdiff {} = {} -----'.format(iteration,maxdiff))
    iteration += 1
    
    if maxdiff < EPS:
        break
        
        
                
    
    