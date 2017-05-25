#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:14:07 2017

@author: lochappy
"""
import grid_world
import numpy as np

#discount factor
alpha = 0.1
gamma = 0.9
EPS = 0.0001 

ALL_POSSIBLE_ACTIONS = ('U','R','L','D')

def random_action(a, eps=0.5):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

#playgame
def playGame(grid,policy):
    
    s = (2,0) #randomly select the start state    
    grid.set_state(s)    
    rewards = [(s,0)]
    while(not grid.game_over()):
        a = random_action(policy[s])            
        r = grid.move(a)
        s = grid.current_state()
        rewards.append((s,r)) 
#        if grid.current_state() == s:
#            rewards.append((s,-100))
#        else:
#            s = grid.current_state()
#            rewards.append((s,r))    
    #print 'rewards=',rewards
    return rewards


############### Monte Carlo Policy Evaluation #####################

#create the standart grid world
g = grid_world.negative_grid(step_cost=-0.1)
#g = grid_world.standard_grid()

# state -> action
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

print "======== Rewards ==========="
grid_world.print_values(g.rewards,g)

#retrieve all the available states in g 
available_actions = g.actions

#retrieve all the available states in g 
available_states = g.all_states()

#g.set_state((0,0))
#g.move('U')
#print g.current_state()
#g.game_over()
# randomly initialize the value of each state
# The values of terminal state must be 0
value_state = {s:0 for s in available_states}

print "======== Initial Value ==========="
grid_world.print_values(value_state,g)

num_episo = 1000
for i in xrange(num_episo):
    states_rewards = playGame(g,policy)
    
    for idx in xrange(len(states_rewards[:-1])):#we dont care the terminal state, cuz by definition, it is 0
        s0, _ = states_rewards[idx]
        s1, r = states_rewards[idx + 1]
        value_state[s0] = value_state[s0] + alpha*(r + gamma*value_state[s1] - value_state[s0]) 
        
print "======== Policy ==========="
grid_world.print_policy(policy,g)              
print "========State value ==========="
grid_world.print_values(value_state,g)







        
        
                
    
    