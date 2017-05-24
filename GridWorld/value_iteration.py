#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:14:07 2017

@author: lochappy
"""
import grid_world
import numpy as np

#create the standart grid world
g = grid_world.negative_grid(step_cost=-0.1)
#g = grid_world.standard_grid()

#retrieve all the available states in g 
available_states = g.all_states()
print available_states

#discount factor
gamma = 0.9
EPS = 0.0001 

#retrieve all the available states in g 
available_actions = g.actions
print available_actions

state_transition_prob = 1 # p(s',r|s,a) = 1 this is the deterministic case
                             
# randomly initialize the value of each state
# The values of terminal state must be 0
value_state = {s:np.random.randn() for s in available_states}
for s in available_states:
    if s not in available_actions: # this is the terminal state
      value_state[s] = 0
             
#### initialize the policy ####
policy = { s: None for s, actions in available_actions.iteritems()}
#grid_world.print_policy(policy, g)

#value iteration and evaluation
iteration = 0
p_a = 1.
# repeat until convergence
# V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
while(1):
    print('---- Iteration {} -----'.format(iteration))
    maxdiff = 0.0
    for s in available_states:
        if s in policy:            
            v_old = value_state[s]
            #iterate over the available actions at the stage s, compute the max_v value
            max_v = float('-inf')
            for action in available_actions[s]:
                g.set_state(s)
                reward = g.move(action)
                new_s = g.current_state()
                v = p_a*state_transition_prob*(reward + gamma*value_state[new_s])
                if v > max_v:
                    max_v = v
            value_state[s]  = max_v
            maxdiff = max(maxdiff,abs(value_state[s] - v_old))
    
    grid_world.print_values(value_state,g)
    print('---- Maxdiff {} = {} -----'.format(iteration,maxdiff))
    iteration += 1
        
    if maxdiff < EPS:
        break
        
    #grid_world.print_values(value_state,g)
    
#compute the final best policy
for s in available_states:
    if s in policy:
        max_Q =  float('-inf')
        #iterate over the available actions at the stage s, compute the q value
        for action in available_actions[s]:
            g.set_state(s)
            r = g.move(action)
            s_prime = g.current_state()
            new_Q = p_a*state_transition_prob*(r + gamma*value_state[s_prime])
                
            if new_Q > max_Q:
                max_Q = new_Q
                policy[s] = action


grid_world.print_policy(policy, g)       










        
        
                
    
    