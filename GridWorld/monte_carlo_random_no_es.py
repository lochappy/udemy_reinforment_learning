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

def random_action(a,eps = 0.5):
    p = np.random.random()
    if p < (1. - eps):
        return a
    else:
        return np.random.choice(list(SET_OF_ACTIONS))

#playgame
def playGame(grid,policy):
    
    s = (2,0) #randomly select the start state
    a = random_action(policy[s]) # randomly select the first action
    
    grid.set_state(s)    
    rewards = [(s,a,0)]
    first = True
    while(not grid.game_over()):
        if (not first):
            a = random_action(policy[s])
        else:
            first = False
        r = grid.move(a)
        if grid.current_state() == s:
            rewards.append((s,a,-100))
            break
        else:
            s = grid.current_state()
            rewards.append((s,a,r))    
    #print 'rewards=',rewards
    returned_s_a_g = []    
    #reverse the sequence from end to start
    rewards.reverse()
    G = 0.#first G(t+1) is the reward of the terminal state, which is 0.0 by definition
    for s,a,r in rewards:
        returned_s_a_g.append((s,a,G))
        G = r + gamma*G # G(t) = r + gamma*G(t+1)
    #print 's_a_g  = ',returned_s_a_g
    returned_s_a_g.reverse()
    
    
    #cuz the first state doest not move at all G is not valid
    #returned_s_g = returned_s_g[:-1]
    
    return returned_s_a_g


############### Monte Carlo Policy Evaluation #####################

#create the standart grid world
g = grid_world.negative_grid(step_cost=-0.1)
#g = grid_world.standard_grid()

# state -> action
# initialize a random policy
policy = {}
for s in g.actions.keys():
    policy[s] = np.random.choice(list(SET_OF_ACTIONS))

print "======== Policy ==========="
grid_world.print_policy(policy,g)

print "======== Rewards ==========="
grid_world.print_rewards(g)

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

all_G = {}
all_returns = {}
for s in available_states:
    for a in SET_OF_ACTIONS:
        all_G[(s,a)] = 0.
        all_returns[(s,a)] = []
num_episo = 10000
for i in xrange(num_episo):
    seen_s = set()
    returns = playGame(grid=g,policy=policy)
    for s,a,r in returns:
        if (s,a) not in seen_s:
            seen_s.add((s,a))
            all_returns[(s,a)].append(r)
            all_G[(s,a)] = np.mean(all_returns[(s,a)])
            
for s in policy.keys():
    maxG = float('-inf')
    maxA = []
    for a in list(SET_OF_ACTIONS):
        if all_G[(s,a)] > maxG:
            maxG = all_G[(s,a)]
            maxA = a
    policy[s] = maxA
     
              
print "======== Optimal Policy ==========="
grid_world.print_policy(policy,g)







        
        
                
    
    