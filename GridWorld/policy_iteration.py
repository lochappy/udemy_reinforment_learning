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
             
#### randomly initialize the policy ####
policy = { s: np.random.choice(actions) for s, actions in available_actions.iteritems()}
#grid_world.print_policy(policy, g)

while(1):
    #policy evaluation
    iteration = 0
    p_a = 1.
    while(1):
        #print('---- Iteration {} -----'.format(iteration))
        maxdiff = 0.0
        for s in available_states:
            if s in policy:
                
                v_old = value_state[s]
                g.set_state(s)
                reward = g.move(policy[s])
                new_s = g.current_state()
                value_state[s] = p_a*state_transition_prob*(reward + gamma*value_state[new_s])
                
                maxdiff = max(maxdiff,abs(value_state[s] - v_old))
    
        #grid_world.print_values(value_state,g)
        #print('---- Maxdiff {} = {} -----'.format(iteration,maxdiff))
        iteration += 1
        
        if maxdiff < EPS:
            break
        
    #grid_world.print_values(value_state,g)
    
    #policy improvement
    policy_has_changed = False
    for s in available_states:
        if s in policy:
            old_action = policy[s]
            old_Q = value_state[s]
            #iterate over the available actions at the stage s, compute the q value
            for action in available_actions[s]:
                g.set_state(s)
                r = g.move(action)
                s_prime = g.current_state()
                new_Q = p_a*state_transition_prob*(r + gamma*value_state[s_prime])
                
                if new_Q > old_Q:
                    old_Q = new_Q
                    policy[s] = action
                
                if policy[s] != old_action:
                    policy_has_changed = True
                    
    if not policy_has_changed:
        break

grid_world.print_policy(policy, g)       










        
        
                
    
    