#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:14:07 2017

@author: lochappy
"""
import grid_world
import numpy as np
import matplotlib.pyplot as plt

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

#create the standart grid world
g = grid_world.negative_grid(step_cost=-0.1)
#g = grid_world.standard_grid()

def maxAction(Q):
    maxA = 'L'
    maxValue = float('-inf')
    for a,v in Q.iteritems():
        if v > maxValue:
            maxValue = v
            maxA = a
            
    return maxA
    
#initialize the Q(s,a)
Q = {}
for s in g.all_states():
    Q[s] = {}
    for a in list(ALL_POSSIBLE_ACTIONS):
        Q[s][a] = 0.

num_episo = 50000
maxQdiff = []
t = 1.0
for i in xrange(num_episo):
    if i % 2000 == 0:
        print "it:", i
        
    if i % 100 == 0:
        t += 0.5
      
    s = (2,0)
    g.set_state(s)
    
    a = maxAction(Q[s])
    a = random_action(a,eps=0.5/t)
    biggest_change = 0
    while(not g.game_over()):
        
        r = g.move(a)
        
        s1 = g.current_state()
        if g.game_over():
            a1 = 'R'
        else:
            a1 = maxAction(Q[s1])
            a1 = random_action(a1,eps=0.5/t)
        
        oldQ = Q[s][a]
        Q[s][a] = Q[s][a] + alpha*(r + gamma*Q[s1][a1] - Q[s][a])#=> import update
        
        biggest_change = max( np.abs(Q[s][a] - oldQ),  biggest_change)
        
        s = s1
        a = a1
        
    maxQdiff.append(biggest_change)

plt.plot(maxQdiff)
plt.show()

#find the final V(s)
V={}
for s in g.actions.keys():
    sumQ = 0
    for a in list(ALL_POSSIBLE_ACTIONS):
        sumQ += Q[s][a]
    V[s] = sumQ / len(ALL_POSSIBLE_ACTIONS)
print "======== V ==========="
grid_world.print_values(V,g)

#find the best policy base on Q(s,a)
policy = {}
for s in g.actions.keys():
    maxQ = float('-inf')
    maxA = []
    for a in list(ALL_POSSIBLE_ACTIONS):
        if Q[s][a] > maxQ:
            maxQ = Q[s][a]
            maxA = a
    policy[s] = maxA

print "======== Optimal Policy ==========="
grid_world.print_policy(policy,g)







        
        
                
    
    