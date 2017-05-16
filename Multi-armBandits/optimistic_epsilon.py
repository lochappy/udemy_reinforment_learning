#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:42:02 2017

@author: lochappy
"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,m):
        self.m = m
        self.Mean = 10.
        self.N = 0.
        
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self,x):
        self.N += 1.
        self.Mean = (1. - 1./self.N)*self.Mean + 1./self.N*x
        
def run_experiment(m1, m2, m3, eps, N, visualization=False):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    
    data = np.empty(N)
    
    for i in xrange(N):
        #optimistic mean
        j = np.argmax([b.Mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        #for visualization
        data[i] = x
        
    cumulative_avarage = np.cumsum(data) / (np.arange(N) + 1.)
    
    #visualization
    if visualization:
        plt.plot(cumulative_avarage)
        plt.plot(np.ones(N)*m1)
        plt.plot(np.ones(N)*m2)
        plt.plot(np.ones(N)*m3)
        plt.xscale('log')   

    print [b.Mean for b in bandits]
    return cumulative_avarage


c_10 = run_experiment(1.,2.,3.,0.1,100000)
c_05 = run_experiment(1.,2.,3.,0.05,100000)  
c_01 = run_experiment(1.,2.,3.,0.01,100000)

plt.plot(c_10,label = 'eps = 0.1')
plt.plot(c_05,label = 'eps = 0.05') 
plt.plot(c_01,label = 'eps = 0.01')
plt.legend() 
plt.xscale('log') 