#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:09:18 2023

@author: john
"""
import random
import numpy as np

##############################################################
#                                                            #   
#                MODEL-BASED ALGORITHMS                      #
#                                                            #
##############################################################       
                                                    
def initialize(env):
    V      = np.zeros(env.num_states)
    policy = np.random.randint(env.num_actions, size=env.num_states)
    return V, policy


def evaluate_policy(V, tol, env, gamma, policy, Vl):
    #V      = np.zeros(env.num_states)
    while True:
        delta = 0
        for s in range(env.num_states):
            a           = policy[s]
            v           = 0
            for s1 in range(env.num_states):
                v      += env.p(s1, s, a)*(env.r(s,a) + gamma*V[s1])
            
            delta       = max(delta, np.abs(v - V[s]))
            V[s]        = v
        Vl.append(np.mean(V))
        if delta < tol:
            break
    return V
    
      
def improve_policy(env, V, policy, gamma, opt):
    policy_stable  = True
    A              = np.arange(env.num_actions)
    for s in range(env.num_states):
        old_action = policy[s]
        Q_a        = np.zeros(env.num_actions)
        
        for a in A:
            for s1 in range(env.num_states):
                Q_a[a]    += env.p(s1, s, a) \
                            *(env.r(s ,a) + gamma*V[s1])
            
        policy[s] = np.argmax(Q_a)
        if old_action != policy[s]:
            policy_stable = False    
    if policy_stable:
        opt = True
        
    return policy, opt
    
        
def value_iteration(env, gamma, tol, max_iterations, Vl):
    V, policy    = initialize(env)
    A            = np.arange(env.num_actions)
    for i in range(max_iterations):
        delta    = 0
        for s in range(env.num_states):
            Q_a         = np.zeros(env.num_actions)
            for a in A:
                for s1 in range(env.num_states):
                    Q_a[a]      += env.p(s1, s, a)* \
                             (env.r(s,a) + gamma*V[s1])
            
            v           = np.max(Q_a)
            policy[s]   = np.argmax(Q_a)
            delta       = max(delta, np.abs(v - V[s]))
            V[s]        = v
        Vl.append(np.mean(V))
        if delta < tol: 
            break   
    return V, policy       
   
   
def policy_iteration(env, gamma, tol, max_iterations, Vl):
    
    V, policy = initialize(env)
    opt       = False
   
    for i in range(max_iterations):
        V    = evaluate_policy(V, tol, env, gamma, policy, Vl)
        policy, opt = improve_policy(env, V, policy, gamma, opt)
        
        #print(i)
        if opt:
            print("Converged.")
            break
    return (V, policy)
    
##############################################################
#                                                            #   
#                 MODEL-FREE ALGORITHMS                      #   
#                                                            #       
##############################################################


def epsilon_greedy(env, Q, s, epsilon):
    a = 0
    if np.random.uniform(0, 1) < epsilon:
        a = random.randrange(env.num_actions)
    else:
        a = np.argmax(Q[s,:])
    return a



def TD0(env, policy, alpha, gamma, total_episodes):
    V = np.zeros(env.num_states)
    for eps in range(total_episodes):
        s = env.reset()
        t = 0
        while t < env.max_num_steps:
            a       = policy[s]
            s1, reward, done = env.step(a)
            target  = reward + gamma*V[s1]
            V[s]    += alpha * (target - V[s])
            s       = s1
            t       += 1
            if done:
                break
    return V


def SARSA(env, total_episodes, epsilon, decay, alpha, gamma, Q_list, G_list):
    Q = np.zeros((env.num_states, env.num_actions))
    for eps in range(total_episodes):
        s       = env.reset()
        G       = 0
        if decay:
            epsilon = 1
        t       = 0
        a       = epsilon_greedy(env, Q, s, epsilon)
        while t < env.max_num_steps:
            if decay:
                epsilon = 1/(t+1)
            s1, reward, done = env.step(a)
            G  += (gamma**t) * reward
            #Choosing the next action
            a1 = epsilon_greedy(env, Q, s1, epsilon)
             
            #Learning the Q-value
            target  = reward + gamma * Q[s1, a1]
            Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
     
            s = s1
            a = a1
           
            t += 1
             
            if done:
                break
        G_list.append(G)
        Q_list.append(np.mean(np.max(Q, axis = 1)))  
    return Q, Q_list, G_list
    



def Q_learning(env, total_episodes, epsilon, decay, alpha, gamma, Q_list, G_list):
    Q = np.zeros((env.num_states, env.num_actions))
    for eps in range(total_episodes):
        s       = env.reset()
        G       = 0
        if decay:
            epsilon = 1
        t       = 0
        a       = epsilon_greedy(env, Q, s, epsilon)
        while t < env.max_num_steps:
            if decay:
                epsilon = 1/(t+1)
            a       = epsilon_greedy(env, Q, s, epsilon)
            s1, reward, done = env.step(a)
            G += (gamma**t) * reward
            #Learning the Q-value
            target  = reward + gamma * np.max(Q[s1, :])
            Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
     
            s = s1
           
            t += 1
             
            if done:
                break
        G_list.append(G)
        Q_list.append(np.mean(np.max(Q, axis = 1)))
    return Q, Q_list, G_list
