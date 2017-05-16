#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:52:13 2017

@author: lochappy
"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,m):
        self.m = m
        self.Mean = 0.
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
        #ubc1
        j = np.argmax([b.Mean + np.sqrt(2*np.log(i)/(b.N + 1e-6)) for b in bandits])
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


c = run_experiment(1.,2.,3.,0.1,100000)

plt.plot(c,label = 'UBC1')
plt.legend() 
plt.xscale('log') 