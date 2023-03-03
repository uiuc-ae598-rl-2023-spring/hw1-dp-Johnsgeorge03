#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:26:43 2023

@author: john
"""

import matplotlib.pyplot as plt
import numpy as np
import algorithms


def plot_q_vs_episodes(Q_list, algo_name, x_label , y_label, title, color):
    eps = np.arange(len(Q_list)) + 1
    plt.figure(figsize=(10, 8))
    plt.plot(eps, Q_list, color, label = algo_name)
    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    plt.title(title, fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()
    

def plot_val_q_vs_episodes(Q_list, value, val_name, ax):
    eps = np.arange(len(Q_list)) + 1
    ax.plot(eps, Q_list, label = val_name + "{}".format(value))
    
    
def plot_trajectory(env, policy):
    
    # Simulate until episode is done
    s     = env.reset()
    done  = False
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    while not done:
        a            = policy[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return log