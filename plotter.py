#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:26:43 2023

@author: john
"""

import matplotlib.pyplot as plt
import numpy as np
import algorithms


def plot_q_vs_episodes(Q_list, algo_name, x_label , y_label, title, color, filepath):
    eps = np.arange(len(Q_list)) + 1
    plt.figure(figsize=(10, 8))
    plt.plot(eps, Q_list, color, label = algo_name)
    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    plt.title(title, fontsize = 20)
    plt.grid()
    plt.legend()
    plt.savefig(filepath, dpi = 100)
    plt.show()
    

def plot_val_q_vs_episodes(Q_list, value, val_name, ax):
    eps = np.arange(len(Q_list)) + 1
    ax.plot(eps, Q_list, label = val_name + "{}".format(value))
    
    
def plot_trajectory(env, policy, pendulum):
    
    # Simulate until episode is done
    s     = env.reset()
    done  = False
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    if pendulum:
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'theta': [env.x[0]],
            'thetadot': [env.x[1]]
        }
        
    if pendulum:
        while not done:
            a            = policy[s]
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(env.x[0])
            log['thetadot'].append(env.x[1])
        
    else:
        
        while not done:
            a            = policy[s]
            (s, r, done) = env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            
    return log